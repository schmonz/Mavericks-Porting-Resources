/*
 * reloc.c — block-window relocation (Task A: relocation skeleton).
 *
 * The eager trampoliner (tramp.c) redirects a run of faulting instructions with
 * a 5-byte `jmp rel32`. An isolated 4-byte scalar BMI op (lzcnt/tzcnt, F3 0F
 * BD/BC) has no room for that jmp, so it keeps trapping at ~57us per SIGILL
 * round-trip forever.
 *
 * avxemu_relocate_block() fixes that by relocating a *window* — the faulting
 * instruction plus enough following position-independent instructions to reach
 * >=5 bytes — into the RWX code cache, then patching the site with a 5-byte jmp:
 *
 *     site:  E9 rel32  ----------------------------+
 *                                                  |
 *     pool:  [ tt spill -> avxemu_emulate(fault) -> reload -> jmp tail ]  <- stub
 *            [ <legal insns copied verbatim> ; E9 rel32 -> end ]          <- tail
 *
 * The faulting instruction runs through the EXISTING emulator (avxemu_emulate,
 * via the tt thunk template + avxemu_tramp_dispatch). No new emulation logic and
 * no native lowering here — that is Task B. The legal following instructions run
 * natively from the pool, then control jumps back to `end` in the original code.
 *
 * Correctness floor: if a safe relocation cannot be PROVEN, return 0 so the
 * caller keeps using the SIGILL emulation path (no regression).
 */
#include "regfile.h"
#include "decode.h"
#include "lde.h"
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <mach/mach.h>
#include <cpuid.h>

/* Must match tramp.c / tramp.s exactly (the run_record is appended after the
 * tt thunk code; the dispatcher walks insns[i].dec / .addr). */
typedef struct { uint64_t addr; decoded dec; } tramp_insn;
typedef struct { uint32_t n; uint32_t pad; tramp_insn insns[]; } run_record;

/* tt thunk template (tramp.s): full spill (all ymm) -> call dispatch -> reload
 * -> jmp resume. Position-independent; the builder memcpy's [tt_start,tt_record),
 * sets the two pointer slots, and appends the run_record. */
extern char avxemu_tt_start[], avxemu_tt_record[];
extern char avxemu_tt_dispatchptr[], avxemu_tt_resumeptr[];
extern void avxemu_tramp_dispatch(const void *record, void *regfile);

/* shared RWX thunk pool (tramp.c) */
extern int    avxemu_pool_init(void *hint, size_t cap);
extern void  *avxemu_pool_base(void);
extern size_t avxemu_pool_used(void);

/* ---- fault classification (mirrors tramp.c's static tramp_faults/detect_features;
 * those are file-static there, so the predicate is replicated here). ---------- */
static int g_lack_avx2 = 1, g_lack_fma = 1, g_lack_bmi = 1, g_lack_f16c = 0, g_det = 0;

static void detect_features(void) {
    if (getenv("AVXEMU_FORCETRAMP")) {     /* dev: treat everything emulatable as faulting */
        g_lack_avx2 = g_lack_fma = g_lack_bmi = g_lack_f16c = 1; g_det = 1; return;
    }
    unsigned a, b, c, d;
    if (__get_cpuid(1, &a, &b, &c, &d)) { g_lack_fma = !(c & (1u<<12)); g_lack_f16c = !(c & (1u<<29)); }
    unsigned a7 = 0, b7 = 0, c7 = 0, d7 = 0;
    __cpuid_count(7, 0, a7, b7, c7, d7);
    g_lack_avx2 = !(b7 & (1u<<5));
    int bmi1 = (b7 & (1u<<3)) != 0, bmi2 = (b7 & (1u<<8)) != 0;
    unsigned a8, b8, c8, d8; int lz = __get_cpuid(0x80000001u, &a8, &b8, &c8, &d8) && (c8 & (1u<<5));
    g_lack_bmi = !(bmi1 && bmi2 && lz);
    g_det = 1;
}
static int avx2_only_op(vex_op op) {
    switch (op) {
    case VPSLLVD: case VPSLLVQ: case VPSRLVD: case VPSRLVQ: case VPSRAVD:
    case VPBROADCASTB: case VPBROADCASTW: case VPBROADCASTD: case VPBROADCASTQ:
    case VBROADCASTI128: case VPBLENDD:
    case VEXTRACTI128: case VINSERTI128: case VPERM2I128:
    case VPERMQ: case VPERMD: case VPERMPD: case VPERMPS: return 1;
    default: return 0;
    }
}
static int block_faults(const decoded *d) {
    if (!g_det) detect_features();
    if (d->is_bmi) return g_lack_bmi;
    if (d->op >= VFMADD132 && d->op <= VFNMSUB231) return g_lack_fma;
    if (d->op == VCVTPH2PS) return g_lack_f16c;
    if (avx2_only_op(d->op)) return g_lack_avx2;
    return d->wide && g_lack_avx2;
}

/* ---- copyability: a post-faulting instruction may be relocated verbatim only
 * if it is position-INDEPENDENT (its bytes mean the same thing executed from the
 * pool as from the original site).
 *
 * Completeness — why the reject-set below is a *complete* floor, not a blacklist:
 * on x86-64 an instruction's behavior depends on its own address ONLY through
 *   (a) a RIP-relative memory operand (ModRM mod!=11 with rm==101, and more
 *       generally any memory operand), and
 *   (b) a relative control transfer (target/return computed as rip + rel).
 * We reject (a) wholesale by rejecting every memory operand (mod != 11b below),
 * which subsumes RIP-relative. We reject (b) completely:
 *   - EB/E9 jmp, Jcc (70-7F, 0F 80-8F), ret (C2/C3/CA/CB), hlt/ud2, and the
 *     indirect jmp (FF /4,/5) are all flagged by lde_cflow -> term/tgt/ind here;
 *   - E8 (call rel32) and E0-E3 (loopne/loope/loop/jrcxz) are NOT flagged by
 *     lde_cflow (call has a fallthrough; loop is not in its branch set), so we
 *     reject them EXPLICITLY below. (call rel32 would also push a pool-relative
 *     return address.)
 * Everything else is RIP-invariant. In particular an INDIRECT call (FF /2, e.g.
 * `call *%rax`) is position-INDEPENDENT and correctly remains copyable: it pushes
 * a return address inside the pool, but that address flows back through the
 * relocated stream to the jmp-back, so control returns correctly.
 *
 * has-ModRM tables mirror lde.c's (file-static there). Anything we cannot fully
 * classify (VEX/EVEX, 0F38/0F3A, 3DNow!) is rejected — the safe floor. ------- */
static int modrm_1(uint8_t op) {
    switch (op) {
    case 0x00:case 0x01:case 0x02:case 0x03: case 0x08:case 0x09:case 0x0A:case 0x0B:
    case 0x10:case 0x11:case 0x12:case 0x13: case 0x18:case 0x19:case 0x1A:case 0x1B:
    case 0x20:case 0x21:case 0x22:case 0x23: case 0x28:case 0x29:case 0x2A:case 0x2B:
    case 0x30:case 0x31:case 0x32:case 0x33: case 0x38:case 0x39:case 0x3A:case 0x3B:
    case 0x63: case 0x69:case 0x6B:
    case 0x80:case 0x81:case 0x83: case 0x84:case 0x85:case 0x86:case 0x87:
    case 0x88:case 0x89:case 0x8A:case 0x8B:case 0x8C:case 0x8D:case 0x8E:case 0x8F:
    case 0xC0:case 0xC1:case 0xC6:case 0xC7:
    case 0xD0:case 0xD1:case 0xD2:case 0xD3:
    case 0xD8:case 0xD9:case 0xDA:case 0xDB:case 0xDC:case 0xDD:case 0xDE:case 0xDF:
    case 0xF6:case 0xF7:case 0xFE:case 0xFF:
        return 1;
    default: return 0;
    }
}
static int modrm_0f(uint8_t op) {
    switch (op) {
    case 0x05:case 0x06:case 0x07:case 0x08:case 0x09:case 0x0B:case 0x0E:
    case 0x30:case 0x31:case 0x32:case 0x33:case 0x34:case 0x35:case 0x37:case 0x77:
    case 0x80:case 0x81:case 0x82:case 0x83:case 0x84:case 0x85:case 0x86:case 0x87:
    case 0x88:case 0x89:case 0x8A:case 0x8B:case 0x8C:case 0x8D:case 0x8E:case 0x8F:
    case 0xA0:case 0xA1:case 0xA2:case 0xA8:case 0xA9:case 0xAA:
    case 0xC8:case 0xC9:case 0xCA:case 0xCB:case 0xCC:case 0xCD:case 0xCE:case 0xCF:
        return 0;
    default: return 1;
    }
}
static int safe_copyable(const uint8_t *p, const uint8_t *end, int len) {
    int term; long tgt; int ind;
    lde_cflow(p, end, len, 0, &term, &tgt, &ind);
    if (term || tgt >= 0 || ind) return 0;            /* any branch/ret/call/jmp */

    const uint8_t *q = p;
    while (q < end) {
        uint8_t c = *q;
        if (c==0x66||c==0x67||c==0xF0||c==0xF2||c==0xF3||c==0x2E||c==0x36||c==0x3E||c==0x26||c==0x64||c==0x65) q++;
        else break;
    }
    if (q >= end) return 0;
    if (*q == 0xC4 || *q == 0xC5 || *q == 0x62) return 0;   /* VEX/EVEX: reject */
    if ((*q & 0xF0) == 0x40) q++;                            /* REX */
    if (q >= end) return 0;
    uint8_t op = *q++;
    /* relative control transfers not caught by lde_cflow (see completeness note
     * above): call rel32 and the loop/jrcxz family are position-DEPENDENT — their
     * rel target (and call's pushed return address) would be computed from the
     * pool, not the site. Reject so the caller falls back to SIGILL emulation. */
    if (op == 0xE8) return 0;                                /* call rel32 */
    if (op == 0xE0 || op == 0xE1 || op == 0xE2 || op == 0xE3) return 0; /* loopne/loope/loop/jrcxz */
    int has_modrm;
    if (op == 0x0F) {
        if (q >= end) return 0;
        uint8_t o2 = *q++;
        if (o2 == 0x38 || o2 == 0x3A || o2 == 0x0F) return 0;   /* 3-byte map / 3DNow! */
        has_modrm = modrm_0f(o2);
    } else {
        has_modrm = modrm_1(op);
    }
    if (has_modrm) {
        if (q >= end) return 0;
        if ((*q >> 6) != 3) return 0;     /* mod != 11b => memory operand (incl. rip-rel) */
    }
    return 1;
}

/* ---- pool allocation. Task C: both the eager trampoliner (constructor) and this
 * runtime relocator allocate from the SAME RWX pool, so there is exactly ONE bump
 * cursor — avxemu_pool_alloc (tramp.c) over g_pool/g_used. We no longer keep a
 * private cursor (which could hand out memory that overlaps the trampoliner's);
 * every allocation here goes through avxemu_pool_alloc.
 *
 * The constructor's install_trampolines normally creates the pool before anything
 * faults. If nothing faulted at load (pool never made) and we are reached anyway,
 * lazy-init it near `site` so jmp rel32 from the patch site stays in range. */
#define POOL_CAP (96u << 20)   /* keep in sync with tramp.c pool_sz */

static int ensure_pool(uint8_t *site) {
    if (avxemu_pool_base()) return 1;                  /* already created by the constructor */
    void *hint = (void *)((((uintptr_t)site) + 0x100000) & ~(uintptr_t)0xfff);
    return avxemu_pool_init(hint, POOL_CAP);
}

static int in_i32(int64_t v) { return v >= INT32_MIN && v <= INT32_MAX; }

/* Task B: count of blocks emitted via an inline native lowering (vs the emulator
 * stub). Test hook (reloctest.c) — proves the inline path actually ran. */
int avxemu_reloc_inlined = 0;

/* ===========================================================================
 * Task B: inline native lowerings for the 4 measured hot scalar BMI ops
 * (LZCNT/TZCNT/SHLX/ANDN). Instead of routing the faulting op through the
 * spill->avxemu_emulate->reload stub (a C call per execution), we emit a SHORT
 * base-ISA byte sequence directly into the pool so the relocated block runs at
 * native speed.
 *
 * The target is a no-AVX2/no-BMI Ivy Bridge: it has bsr/bsf/SSE4.2 but NOT
 * lzcnt/tzcnt/shlx/andn (those #UD). So the lowerings use ONLY base-ISA
 * instructions (bsr/bsf/shl-by-cl/and/not/cmov/test/setcc/...). reloc.c is part
 * of the SSE-only core: we WRITE machine-code bytes (data) into the pool, which
 * is fine — the compiled reloc.c itself emits no AVX/VEX.
 *
 * Semantics are bit-exact to bmi_exec (exec_bmi.c), INCLUDING flags. Critically,
 * set_flags() in exec_bmi.c writes ONLY CF/ZF/SF/OF and leaves PF/AF untouched,
 * so every lowering PRESERVES PF/AF (and all unowned flags). We achieve exact
 * flags by bracketing with pushfq...popfq and patching only the owned bits
 * (CF|ZF|SF|OF) of the saved flags image on the stack:
 *   - LZCNT/TZCNT: CF=(src==0), ZF=(result==0), SF=0, OF=0.
 *   - ANDN:        CF=0, OF=0, ZF=(r==0), SF=sign(r).
 *   - SHLX:        writes NO flags (pushfq/popfq preserves all 6).
 * 32-bit ops operate at 32-bit width so the result zero-extends the dest to 64
 * and the source high halves are ignored (matching bmi_exec's maskz).
 *
 * Any op/operand shape not handled here (memory operand, rsp operand, MULX,
 * opsize!=32/64) returns 0 from emit_lowering -> caller falls back to the stub.
 * =========================================================================== */
#define RSP 4
static void eb(uint8_t **p, uint8_t b){ *(*p)++ = b; }
static void e32(uint8_t **p, uint32_t v){ memcpy(*p, &v, 4); *p += 4; }
static uint8_t modrm_rr(int reg, int rm){ return (uint8_t)(0xC0 | ((reg&7)<<3) | (rm&7)); }
static void rex(uint8_t **p, int w, int reg, int rm, int force){
    uint8_t r = (uint8_t)((w?8:0) | ((reg>=8)?4:0) | ((rm>=8)?1:0));
    if (r || force) eb(p, (uint8_t)(0x40 | r));
}
/* reg/reg, 1-byte opcode (ModRM.reg=reg, ModRM.rm=rm) */
static void alu_rr(uint8_t **p, int w, uint8_t opc, int reg, int rm){
    rex(p, w, reg, rm, 0); eb(p, opc); eb(p, modrm_rr(reg, rm));
}
/* reg/reg, 0F-escape opcode */
static void alu0f_rr(uint8_t **p, int w, uint8_t opc, int reg, int rm){
    rex(p, w, reg, rm, 0); eb(p, 0x0F); eb(p, opc); eb(p, modrm_rr(reg, rm));
}
static void emit_push(uint8_t **p, int r){ if(r>=8) eb(p,0x41); eb(p,(uint8_t)(0x50|(r&7))); }
static void emit_pop (uint8_t **p, int r){ if(r>=8) eb(p,0x41); eb(p,(uint8_t)(0x58|(r&7))); }
static void emit_mov_imm32(uint8_t **p, int r, uint32_t imm){ /* mov r32,imm32 (zero-extends to r64) */
    if (r>=8) eb(p,0x41); eb(p,(uint8_t)(0xB8|(r&7))); e32(p,imm);
}
static void emit_mov_rr (uint8_t **p,int w,int d,int s){ alu_rr(p,w,0x89,s,d); } /* d = s   (MOV r/m,r) */
static void emit_and_rr (uint8_t **p,int w,int d,int s){ alu_rr(p,w,0x21,s,d); } /* d &= s  (AND r/m,r) */
static void emit_or_rr  (uint8_t **p,int w,int d,int s){ alu_rr(p,w,0x09,s,d); } /* d |= s */
static void emit_sub_rr (uint8_t **p,int w,int d,int s){ alu_rr(p,w,0x29,s,d); } /* d -= s */
static void emit_xor_rr (uint8_t **p,int w,int d,int s){ alu_rr(p,w,0x31,s,d); } /* d ^= s */
static void emit_test_rr(uint8_t **p,int w,int a,int b){ alu_rr(p,w,0x85,a,b); } /* TEST a,b */
static void emit_not    (uint8_t **p,int w,int r){ rex(p,w,0,r,0); eb(p,0xF7); eb(p,(uint8_t)(0xC0|(2<<3)|(r&7))); } /* NOT r (/2) */
static void emit_bsf    (uint8_t **p,int w,int d,int s){ alu0f_rr(p,w,0xBC,d,s); }
static void emit_bsr    (uint8_t **p,int w,int d,int s){ alu0f_rr(p,w,0xBD,d,s); }
static void emit_cmovz  (uint8_t **p,int w,int d,int s){ alu0f_rr(p,w,0x44,d,s); }
/* setcc r8 (cc: 0x94=setz, 0x98=sets). Force a REX byte so r4..r7 select the
 * low byte (spl/bpl/sil/dil) uniformly rather than ah/ch/dh/bh. */
static void emit_setcc  (uint8_t **p,uint8_t cc,int r){
    eb(p,(uint8_t)(0x40 | ((r>=8)?1:0))); eb(p,0x0F); eb(p,cc); eb(p,(uint8_t)(0xC0|(r&7)));
}
static void emit_shl_imm(uint8_t **p,int w,int r,uint8_t i){ rex(p,w,0,r,0); eb(p,0xC1); eb(p,(uint8_t)(0xC0|(4<<3)|(r&7))); eb(p,i); }
static void emit_pushfq(uint8_t **p){ eb(p,0x9C); }
static void emit_popfq (uint8_t **p){ eb(p,0x9D); }
/* shl <opsize> [rsp], cl  (D3 /4 ; SIB base=rsp, no disp) */
static void emit_shl_rsp_cl(uint8_t **p,int w){ if(w) eb(p,0x48); eb(p,0xD3); eb(p,0x24); eb(p,0x24); }
/* and qword [rsp+16], imm32  (81 /4 id) */
static void emit_and_rsp16_imm(uint8_t **p,uint32_t imm){ eb(p,0x48); eb(p,0x81); eb(p,0x64); eb(p,0x24); eb(p,0x10); e32(p,imm); }
/* or  qword [rsp+16], reg    (09 /r) */
static void emit_or_rsp16_reg(uint8_t **p,int r){ eb(p,(uint8_t)(0x48|((r>=8)?4:0))); eb(p,0x09); eb(p,(uint8_t)(0x40|((r&7)<<3)|0x04)); eb(p,0x24); eb(p,0x10); }
/* lea rsp,[rsp+8]  (discard a stack slot; does NOT touch flags, unlike add) */
static void emit_drop8(uint8_t **p){ eb(p,0x48); eb(p,0x8D); eb(p,0x64); eb(p,0x24); eb(p,0x08); }
/* Red-zone bracketing: the inline lowerings pushfq/push starting at the live rsp,
 * which writes into [rsp-8..] — inside the 128-byte System V red zone, corrupting
 * caller locals kept there. Skip the red zone with `lea rsp,[rsp-128]` FIRST and
 * restore with `lea rsp,[rsp+128]` LAST. lea (unlike add/sub) does NOT disturb
 * flags, so it is safe to bracket the pushfq/popfq with these. Net rsp change is
 * zero across the whole lowering, so the verbatim-copied tail sees original rsp.
 *   open:  48 8D 64 24 80          lea rsp,[rsp-128]  (disp8 -128 fits signed)
 *   close: 48 8D A4 24 80 00 00 00 lea rsp,[rsp+128]  (disp32: +128 > disp8 max) */
static void emit_redzone_open (uint8_t **p){ eb(p,0x48); eb(p,0x8D); eb(p,0x64); eb(p,0x24); eb(p,0x80); }
static void emit_redzone_close(uint8_t **p){ eb(p,0x48); eb(p,0x8D); eb(p,0xA4); eb(p,0x24); eb(p,0x80); eb(p,0x00); eb(p,0x00); eb(p,0x00); }

static int gpr_ok(int r){ return r >= 0 && r <= 15 && r != RSP; }
/* lowest two GPRs (!=rsp) not in the avoid set, for scratch */
static int pick_scratch(int a0,int a1,int a2,int got[2]){
    int k=0;
    for (int r=0;r<16 && k<2;r++){
        if (r==RSP || r==a0 || r==a1 || r==a2) continue;
        got[k++]=r;
    }
    return k==2;
}

/* Append a native lowering for the op described by d to *p. Returns 1 on success
 * (p advanced past the emitted bytes), 0 if not inline-lowerable (caller uses the
 * emulator stub). Emits ONLY base-ISA instructions — no LZCNT/TZCNT/BMI/VEX. */
static int emit_lowering(uint8_t **p, const decoded *d){
    if (!d->is_bmi) return 0;
    if (d->dst_kind != DST_GPR) return 0;
    if (!(d->opsize == 32 || d->opsize == 64)) return 0;
    if (d->bmi_dst2 != OPND_NONE) return 0;        /* MULX: two dests, not handled */
    const int w = (d->opsize == 64);
    const uint32_t n = d->opsize;
    const uint32_t CLR = 0xFFFFF73Eu;              /* ~(CF|ZF|SF|OF), sign-extended in AND r/m64,imm32 */

    switch (d->op) {
    case BMI_TZCNT:
    case BMI_LZCNT: {
        int D = d->dst, S = d->a_src;
        if (!gpr_ok(D) || !gpr_ok(S)) return 0;
        int t[2]; if (!pick_scratch(D, S, -1, t)) return 0;
        int T1 = t[0], T2 = t[1];
        emit_redzone_open(p);                                 /* -128: skip red zone */
        emit_pushfq(p); emit_push(p, T1); emit_push(p, T2);   /* saved flags now at [rsp+16] */
        if (d->op == BMI_TZCNT) {
            /* Capture CF=(S==0) BEFORE bsf, since bsf D,S clobbers S when D==S
             * (e.g. tzcnt eax,eax). Mirrors the LZCNT order below. */
            emit_xor_rr(p, 1, T1, T1);
            emit_test_rr(p, w, S, S);          /* ZF=(S==0) at opsize width, S still source */
            emit_setcc(p, 0x94, T1);           /* T1 = (S==0) -> CF (bit0) */
            emit_bsf(p, w, D, S);              /* D = tzcnt(S) if S!=0; ZF=(S==0), D undef if S==0 */
            emit_mov_imm32(p, T2, n);
            emit_cmovz(p, 1, D, T2);           /* S==0 -> D = opsize (reads bsf ZF) */
        } else {                               /* LZCNT */
            emit_xor_rr(p, 1, T1, T1);
            emit_test_rr(p, w, S, S);
            emit_setcc(p, 0x94, T1);           /* T1 = (S==0), captured before S may be clobbered */
            emit_bsr(p, w, D, S);              /* D = index of high set bit (if S!=0) */
            emit_mov_imm32(p, T2, n - 1);
            emit_sub_rr(p, 1, T2, D);          /* T2 = (opsize-1) - bsr  = lzcnt for S!=0 */
            emit_test_rr(p, 1, T1, T1);        /* ZF=(T1==0)=(S!=0) */
            emit_mov_imm32(p, D, n);           /* assume zero-input: D = opsize */
            emit_cmovz(p, 1, D, T2);           /* S!=0 -> D = T2 */
        }
        /* ZF = (D==0) -> T2 bit6; SF/OF stay 0 */
        emit_xor_rr(p, 1, T2, T2);
        emit_test_rr(p, w, D, D);
        emit_setcc(p, 0x94, T2);
        emit_shl_imm(p, 1, T2, 6);
        emit_or_rr(p, 1, T1, T2);              /* owned = CF | ZF */
        emit_and_rsp16_imm(p, CLR);            /* clear CF/ZF/SF/OF in saved flags */
        emit_or_rsp16_reg(p, T1);              /* set computed owned bits */
        emit_pop(p, T2); emit_pop(p, T1); emit_popfq(p);
        emit_redzone_close(p);                 /* +128: restore red zone (net rsp=0) */
        return 1;
    }
    case BMI_ANDN: {
        int D = d->dst, s1 = d->a_src, s2 = d->b_src;
        if (!gpr_ok(D) || !gpr_ok(s1) || !gpr_ok(s2)) return 0;
        int t[2]; if (!pick_scratch(D, s1, s2, t)) return 0;
        int T1 = t[0], T2 = t[1];
        emit_redzone_open(p);                  /* -128: skip red zone */
        emit_pushfq(p); emit_push(p, T1); emit_push(p, T2);
        emit_mov_rr(p, w, T1, s1);
        emit_not(p, w, T1);
        emit_and_rr(p, w, T1, s2);             /* T1 = ~s1 & s2 (opsize width) */
        emit_mov_rr(p, w, D, T1);              /* D = result (zero-extends if 32-bit) */
        emit_xor_rr(p, 1, T1, T1);
        emit_xor_rr(p, 1, T2, T2);
        emit_test_rr(p, w, D, D);              /* ZF=(D==0), SF=sign(D) */
        emit_setcc(p, 0x94, T1);               /* (D==0) */
        emit_setcc(p, 0x98, T2);               /* sign */
        emit_shl_imm(p, 1, T1, 6);             /* -> ZF bit6 */
        emit_shl_imm(p, 1, T2, 7);             /* -> SF bit7 */
        emit_or_rr(p, 1, T1, T2);              /* owned = ZF | SF (CF=0, OF=0) */
        emit_and_rsp16_imm(p, CLR);
        emit_or_rsp16_reg(p, T1);
        emit_pop(p, T2); emit_pop(p, T1); emit_popfq(p);
        emit_redzone_close(p);                 /* +128: restore red zone (net rsp=0) */
        return 1;
    }
    case BMI_SHLX: {
        int D = d->dst, V = d->a_src, C = d->b_src;   /* D = V << (C & (opsize-1)); no flags */
        if (!gpr_ok(D) || !gpr_ok(V) || !gpr_ok(C)) return 0;
        emit_redzone_open(p);                   /* -128: skip red zone */
        emit_pushfq(p);                         /* SHLX writes no flags: save+restore all */
        emit_push(p, 1);                        /* save rcx */
        emit_push(p, V);                        /* push Vval, then Cval, BEFORE rcx changes */
        emit_push(p, C);                        /* (so V/C aliasing rcx reads originals)   */
        emit_pop(p, 1);                         /* rcx = count */
        emit_shl_rsp_cl(p, w);                  /* shift Vval (top slot) by cl (opsize) */
        emit_pop(p, D);                         /* D = shifted value */
        if (D != 1) emit_pop(p, 1);             /* restore rcx */
        else        emit_drop8(p);              /* D==rcx: discard saved rcx (no flag change) */
        if (!w) emit_mov_rr(p, 0, D, D);        /* 32-bit: zero-extend dest upper half */
        emit_popfq(p);
        emit_redzone_close(p);                  /* +128: restore red zone (net rsp=0) */
        return 1;
    }
    default:
        return 0;
    }
}

/* Patch [site,site+5) with `jmp rel32` (writable -> write -> executable).
 *
 * MILESTONE-A LIMITATION (known, deferred): this flips a live __text page RW->RX
 * and writes the 5-byte jmp WHILE the program runs. In a multithreaded target
 * another thread executing this function could observe a half-written jmp or the
 * transient RW page, and corrupt. That is acceptable here because the startup spin
 * this targets is single-threaded main-thread (per the project brief): no other
 * thread is in this code at patch time. Cross-thread-safe live patching
 * (stop-the-world / atomic single-instruction patch) is out of scope for
 * Milestone A and deferred — do NOT rely on this being safe under concurrency. */
static int patch_site_jmp(uint8_t *site, int64_t srel){
    if (!in_i32(srel)) return 0;
    mach_port_t task = mach_task_self();
    uintptr_t lo = (uintptr_t)site & ~(uintptr_t)0xfff;
    uintptr_t hi = ((uintptr_t)site + 5 + 0xfff) & ~(uintptr_t)0xfff;
    if (vm_protect(task, (vm_address_t)lo, (vm_size_t)(hi - lo), FALSE,
                   VM_PROT_READ | VM_PROT_WRITE | VM_PROT_COPY) != KERN_SUCCESS)
        return 0;
    site[0] = 0xE9; { int32_t r32 = (int32_t)srel; memcpy(site + 1, &r32, 4); }
    vm_protect(task, (vm_address_t)lo, (vm_size_t)(hi - lo), FALSE,
               VM_PROT_READ | VM_PROT_EXECUTE);
    return 1;
}

#define WINDOW_LIM 64    /* a window only needs to reach 5 bytes; bound the walk */

/* CALLER PRECONDITION (window safety): patching the site overwrites 5 bytes
 * [site, site+5) with `jmp rel32`. If the faulting instruction is shorter than 5
 * bytes (a 4-byte scalar BMI op leaves [site+4, site+5) inside the footprint),
 * any inbound branch whose target lands strictly inside that footprint would,
 * after patching, decode into the middle of the jmp and corrupt control flow.
 * Task C closes this: avxemu_relocate_block now calls avxemu_patch_safe(site, 5)
 * (tramp.c) before patching, which scans the containing function's branch targets
 * and declines if any lands in the open interval (site, site+5). So the inbound-
 * branch guarantee is enforced here, not pushed onto the caller. */
int avxemu_reloc_last_reason;   /* diagnostic: why the last relocate declined (AVXEMU_FAULTHIST) */
#define RELOC_DECLINE(r) do { avxemu_reloc_last_reason = (r); return 0; } while (0)

int avxemu_relocate_block(uint8_t *site) {
    if (!ensure_pool(site)) RELOC_DECLINE(1);

    /* decode the faulting instruction */
    decoded fd;
    int fl = decode(site, &fd);
    if (fl <= 0 || fd.op == 0) RELOC_DECLINE(2);
    if (!block_faults(&fd)) RELOC_DECLINE(3);

    /* build the window [site,end): walk WHOLE following instructions until >=5
     * bytes; each added instruction must be safely copyable (position-independent) */
    const uint8_t *lim = site + WINDOW_LIM;
    uint8_t *end = site + fl;
    while ((size_t)(end - site) < 5) {
        int zk, off;
        int l = x86_len(end, lim, &zk, &off);
        if (l <= 0) RELOC_DECLINE(4);
        if (!safe_copyable(end, lim, l)) RELOC_DECLINE(4);
        end += l;
        if ((size_t)(end - site) > WINDOW_LIM) RELOC_DECLINE(4);
    }
    /* the 5-byte jmp we write at the site must stay within the relocated window
     * (bytes [site+1,site+5) inside the gathered instruction boundaries) */
    if ((size_t)(end - site) < 5) RELOC_DECLINE(4);

    /* Window patch-safety (Part 3): the 5-byte jmp we write clobbers [site,site+5).
     * If any branch ELSEWHERE in the program targets an address in the open interval
     * (site, site+5) it would, after patching, decode into the middle of the jmp and
     * corrupt that entry path. Decline if we cannot PROVE no such target exists.
     * Declining here (the site keeps faulting/emulating, no corruption) is the safe
     * floor — we never trade correctness for speed. */
    if (!avxemu_patch_safe(site, 5)) RELOC_DECLINE(5);

    size_t legal_len = (size_t)(end - (site + fl));

    /* Task B: prefer an inline native lowering for the faulting op. If one
     * exists, the relocated block is [native lowering][legal tail][jmp back] and
     * runs entirely natively (no C call). Otherwise fall through to the stub. */
    {
        uint8_t lbuf[256];
        uint8_t *lp = lbuf;
        if (emit_lowering(&lp, &fd)) {
            size_t low_len = (size_t)(lp - lbuf);
            uint8_t *blk = avxemu_pool_alloc(low_len + legal_len + 5);
            if (!blk) RELOC_DECLINE(6);
            int64_t srel = (int64_t)(blk - (site + 5));
            if (!in_i32(srel)) RELOC_DECLINE(7);
            memcpy(blk, lbuf, low_len);
            if (legal_len) memcpy(blk + low_len, site + fl, legal_len);
            uint8_t *jb = blk + low_len + legal_len;
            int64_t jrel = (int64_t)((uint8_t *)end - (jb + 5));
            if (!in_i32(jrel)) RELOC_DECLINE(7);
            jb[0] = 0xE9; { int32_t r32 = (int32_t)jrel; memcpy(jb + 1, &r32, 4); }
            if (!patch_site_jmp(site, srel)) RELOC_DECLINE(8);
            avxemu_reloc_inlined++;
            return 1;
        }
    }

    /* allocate stub + tail from the pool (stub is the entry the site jumps to) */
    size_t code  = (size_t)(avxemu_tt_record - avxemu_tt_start);
    size_t recsz = sizeof(run_record) + sizeof(tramp_insn);
    uint8_t *stub = avxemu_pool_alloc(code + recsz);
    uint8_t *tail = avxemu_pool_alloc(legal_len + 5);
    if (!stub || !tail) RELOC_DECLINE(6);

    /* site -> stub must be a reachable rel32 jmp */
    int64_t srel = (int64_t)(stub - (site + 5));
    if (!in_i32(srel)) RELOC_DECLINE(7);

    /* (a) stub: copy the tt template verbatim, wire dispatch + resume, append the
     * 1-instruction run_record for the faulting op (run via avxemu_emulate). */
    memcpy(stub, avxemu_tt_start, code);
    *(void   **)(stub + (avxemu_tt_dispatchptr - avxemu_tt_start)) = (void *)avxemu_tramp_dispatch;
    *(uint64_t *)(stub + (avxemu_tt_resumeptr   - avxemu_tt_start)) = (uint64_t)(uintptr_t)tail;
    run_record *r = (run_record *)(stub + (avxemu_tt_record - avxemu_tt_start));
    r->n = 1; r->pad = 0;
    r->insns[0].addr = (uint64_t)(uintptr_t)site;
    r->insns[0].dec  = fd;

    /* (b) tail: legal instructions verbatim, then jmp back to `end`. */
    if (legal_len) memcpy(tail, site + fl, legal_len);
    uint8_t *jb = tail + legal_len;
    int64_t jrel = (int64_t)((uint8_t *)end - (jb + 5));
    if (!in_i32(jrel)) RELOC_DECLINE(7);
    jb[0] = 0xE9; { int32_t r32 = (int32_t)jrel; memcpy(jb + 1, &r32, 4); }

    /* (c) patch the site with `jmp rel32 -> stub` (writable -> write -> executable). */
    if (!patch_site_jmp(site, srel)) RELOC_DECLINE(8);
    return 1;
}
