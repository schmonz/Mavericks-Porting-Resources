/*
 * tramp.c — trampoline dispatch, thunk pool, and thunk builder.
 *
 * A "run" is a maximal sequence of consecutive faulting instructions. Each run
 * gets one thunk (tramp.s template + an appended run_record). The patched site
 * jumps to the thunk; the thunk snapshots state into a regfile and calls
 * avxemu_tramp_dispatch(), which emulates each instruction of the run in order.
 */
#include "regfile.h"
#include "decode.h"
#include "vexops.h"
#include "lde.h"
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>     /* rename() in ophist_dump (else implicit-decl error on clang>=15) */
#include <sys/mman.h>
#include <pthread.h>
#include <signal.h>
#include <fcntl.h>
#include <mach/mach.h>
#include <mach-o/dyld.h>
#include <mach-o/loader.h>
#include <cpuid.h>

/* run_record: laid out immediately after the thunk code (see tramp.s). */
typedef struct { uint64_t addr; decoded dec; } tramp_insn;   /* addr feeds rf->rip per instruction */
typedef struct { uint32_t n; uint32_t pad; tramp_insn insns[]; } run_record;

extern char avxemu_tt_start[], avxemu_tt_end[];
extern char avxemu_tt_dispatchptr[], avxemu_tt_resumeptr[], avxemu_tt_record[];
/* GPR-only thunk template: saves only xmm0-15 (128-bit) instead of all ymm. */
extern char avxemu_ttg_start[], avxemu_ttg_end[];
extern char avxemu_ttg_dispatchptr[], avxemu_ttg_resumeptr[], avxemu_ttg_record[];
/* BMI reg-only thunk template: saves GPRs+flags only, no vector save at all. */
extern char avxemu_tt2_start[], avxemu_tt2_end[];
extern char avxemu_tt2_dispatchptr[], avxemu_tt2_resumeptr[], avxemu_tt2_record[];

static void emit(const char *s){ (void)write(2, s, strlen(s)); }

/* ============================================================================
 * Milestone-B DIAGNOSTIC (env-gated; harmless when unset): execution-weighted
 * histogram of which ops actually go through C emulation (i.e. are NOT native-
 * codegen'd) during a run. AVXEMU_OPHIST=1 turns it on; dump on SIGUSR2 (so a
 * live spinning process can be probed via `kill -USR2 <pid>`) and atexit. The
 * dump goes to /tmp/ophist.out (parent-readable; the child's stderr is on a pty).
 * ========================================================================== */
static uint64_t g_ophist[VEX_OP_COUNT];
static int g_ophist_on = -1;
static int ophist_enabled(void){
    if (g_ophist_on < 0){ const char *e = getenv("AVXEMU_OPHIST"); g_ophist_on = (e && e[0] && e[0] != '0') ? 1 : 0; }
    return g_ophist_on;
}
/* async-signal-safe unsigned decimal writer */
static void as_u64(int fd, uint64_t v){
    char b[24]; int i = 24;
    if (v == 0){ (void)write(fd, "0", 1); return; }
    while (v){ b[--i] = (char)('0' + (v % 10)); v /= 10; }
    (void)write(fd, b + i, (size_t)(24 - i));
}
/* async-signal-safe: open/write/close + strlen on static strings only, no malloc */
static void ophist_dump_fd(int fd){
    int idx[VEX_OP_COUNT]; int m = 0;
    for (int i = 0; i < VEX_OP_COUNT; i++) if (g_ophist[i]) idx[m++] = i;
    for (int i = 0; i < m; i++){ int best = i;
        for (int j = i + 1; j < m; j++) if (g_ophist[idx[j]] > g_ophist[idx[best]]) best = j;
        int t = idx[i]; idx[i] = idx[best]; idx[best] = t; }
    static const char hdr[] = "=== AVXEMU_OPHIST: C-emulated op histogram (execution-weighted, desc) ===\n";
    (void)write(fd, hdr, sizeof hdr - 1);
    uint64_t total = 0;
    for (int i = 0; i < m; i++){
        const char *nm = vex_op_name((vex_op)idx[i]);
        (void)write(fd, nm, strlen(nm)); (void)write(fd, "\t", 1);
        as_u64(fd, g_ophist[idx[i]]); (void)write(fd, "\n", 1);
        total += g_ophist[idx[i]];
    }
    (void)write(fd, "TOTAL\t", 6); as_u64(fd, total); (void)write(fd, "\n", 1);
}
/* atomic: write to a temp path then rename, so /tmp/ophist.out is never a partial
 * file even if a kill -9 lands mid-dump (rename is atomic). */
static void ophist_dump(void){
    int fd = open("/tmp/ophist.out.tmp", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { ophist_dump_fd(2); return; }
    ophist_dump_fd(fd);
    close(fd);
    (void)rename("/tmp/ophist.out.tmp", "/tmp/ophist.out");
}
static void ophist_sigusr2(int sig){ (void)sig; ophist_dump(); }
/* Robust fallback: the host (Bun/JSC) may reinstall its own SIGUSR2 handler and a
 * kill -9 skips atexit, so also snapshot to /tmp/ophist.out periodically from the
 * hot path. O_TRUNC means the file always holds a recent cumulative snapshot. */
static uint64_t g_ophist_total;
#define OPHIST_SNAP_EVERY (1u << 17)
static inline void ophist_tick(void){
    if (((++g_ophist_total) & (OPHIST_SNAP_EVERY - 1)) == 0) ophist_dump();
}

/* The actual emulation, run on the side stack. */
static void tramp_emulate_run(const void *recordp, void *rfp) {
    const run_record *r = (const run_record *)recordp;
    avxemu_regfile *rf = (avxemu_regfile *)rfp;
    for (uint32_t i = 0; i < r->n; i++) {
        rf->rip = r->insns[i].addr;
        if (g_ophist_on) { g_ophist[r->insns[i].dec.op]++; ophist_tick(); }   /* AVXEMU_OPHIST diagnostic */
        if (!avxemu_emulate(&r->insns[i].dec, rf)) {
            /* Scanner only trampolines ops we emulate, so this is a bug net. */
            emit("avxemu: tramp dispatch: emulate failed mid-run\n");
            return;
        }
    }
}

/*
 * Per-thread side stack for the emulation. A trampolined instruction can fire at
 * any JS recursion depth; running the (multi-KB) emulation on the program stack
 * would add pressure exactly where JSC is near its limit. Moving it to a private
 * stack means a thunk costs only its tiny register-spill frame on the program
 * stack, never the emulation itself.
 */
#define SIDE_SZ (256u * 1024)
extern void avxemu_run_on_stack(uint8_t *base, size_t sz,
                                void (*fn)(const void *, void *), const void *a, void *b);

/* Per-thread side-stack pointer. pthread TSD (not __thread): getspecific is in
 * the baseline system libs, so it is never trampolined and never mallocs — safe
 * to call from a thunk even one that fired inside Bun's lzcnt-using allocator. */
static pthread_key_t g_side_key;
static int g_side_key_ok = 0;

/* Called from the thunk (tramp.s) with the live state spilled into rf. */
void avxemu_tramp_dispatch(const run_record *r, avxemu_regfile *rf) {
    if (!g_side_key_ok) { tramp_emulate_run(r, rf); return; }   /* no key: program stack */
    uint8_t *side = (uint8_t *)pthread_getspecific(g_side_key);
    if (!side) {
        void *m = mmap(0, SIDE_SZ, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (m == MAP_FAILED) { tramp_emulate_run(r, rf); return; }
        side = (uint8_t *)m;
        pthread_setspecific(g_side_key, side);
    }
    avxemu_run_on_stack(side, SIDE_SZ, tramp_emulate_run, r, rf);
}

/* xmm-CLEAN dispatch for register-only BMI runs. Replicates avxemu_emulate's
 * is_bmi path but with NO memory operands (so no mem_read/memcpy) and calls
 * bmi_exec directly -- the whole chain uses only general registers, so it cannot
 * clobber the program's xmm/ymm. That is what lets the tt2 thunk skip the vector
 * save entirely. (A build-time `otool | grep xmm` over this + bmi_exec guards the
 * invariant; the classifier never routes a memory/seg op here.) */
static int emulate_bmi_reg(const decoded *d, avxemu_regfile *rf) {
    uint64_t s1 = 0, s2 = 0, dst = 0, dst2 = 0, flags = rf->rflags;
    if (d->a_src >= 0) s1 = rf->gpr[d->a_src];
    if (d->bmi_s1_rdx) s1 = rf->gpr[2];
    if (d->op == BMI_RORX)  s2 = d->imm;
    else if (d->b_src >= 0) s2 = rf->gpr[d->b_src];
    if (!bmi_exec(d->op, d->opsize, s1, s2, &dst, &dst2, &flags)) return 0;
    /* 16-bit ops (66-prefixed lzcnt/tzcnt/movbe) write only the low word of the
     * destination register on real hardware; 32/64-bit writes zero-extend. */
    if (d->dst >= 0)      rf->gpr[d->dst] = (d->opsize == 16)
                              ? ((rf->gpr[d->dst] & ~0xFFFFull) | (dst & 0xFFFFull))
                              : dst;
    if (d->bmi_dst2 >= 0) rf->gpr[d->bmi_dst2] = dst2;
    rf->rflags = flags;
    return 1;
}
void avxemu_tramp_dispatch_bmi(const run_record *r, avxemu_regfile *rf) {
    for (uint32_t i = 0; i < r->n; i++) {
        rf->rip = r->insns[i].addr;
        if (g_ophist_on) { g_ophist[r->insns[i].dec.op]++; ophist_tick(); }   /* AVXEMU_OPHIST diagnostic */
        if (!emulate_bmi_reg(&r->insns[i].dec, rf)) {
            emit("avxemu: tramp dispatch (bmi): emulate failed mid-run\n");
            return;
        }
    }
}

/* ---- thunk pool: RWX, bump-allocated. The patcher reaches it with jmp rel32,
 * so it must land within +/-2GB of __text; avxemu_pool_init() takes a hint. ---- */
static uint8_t *g_pool; static size_t g_used, g_cap;

int avxemu_pool_init(void *hint, size_t cap) {
    void *p = mmap(hint, cap, PROT_READ | PROT_WRITE | PROT_EXEC,
                   MAP_PRIVATE | MAP_ANON, -1, 0);
    if (p == MAP_FAILED) return 0;
    g_pool = (uint8_t *)p; g_cap = cap; g_used = 0;
    return 1;
}
void *avxemu_pool_base(void){ return g_pool; }
size_t avxemu_pool_used(void){ return g_used; }

/* Raw 16-byte-aligned bump allocation from the shared pool. This is the SINGLE
 * cursor (g_used) over g_pool: the eager trampoliner (build_thunk_t below) and the
 * runtime relocator (reloc.c) both allocate through it, so the two can never hand
 * out overlapping bytes of the live pool. Returns NULL if uninitialized/exhausted. */
void *avxemu_pool_alloc(size_t n) {
    n = (n + 15) & ~(size_t)15;
    if (!g_pool || g_used + n > g_cap) return 0;
    uint8_t *p = g_pool + g_used; g_used += n;
    return p;
}

/*
 * Build a thunk for a run of n decoded instructions (each with its original
 * address) that resumes at `resume`. Returns the thunk entry, or NULL if the
 * pool is exhausted. Only data slots are written; the template code is verbatim.
 */
static void *build_thunk_t(const tramp_insn *insns, int n, uint64_t resume,
                           const char *tstart, const char *trecord,
                           const char *tdispatch, const char *tresume,
                           void *dispatch_fn) {
    size_t code  = (size_t)(trecord - tstart);                     /* code + 2 ptr slots */
    size_t recsz = sizeof(run_record) + (size_t)n * sizeof(tramp_insn);
    uint8_t *t = avxemu_pool_alloc(code + recsz);                   /* shared single cursor */
    if (!t) return 0;
    memcpy(t, tstart, code);
    *(void   **)(t + (tdispatch - tstart)) = dispatch_fn;
    *(uint64_t *)(t + (tresume   - tstart)) = resume;

    run_record *r = (run_record *)(t + (trecord - tstart));
    r->n = (uint32_t)n; r->pad = 0;
    for (int i = 0; i < n; i++) r->insns[i] = insns[i];
    return t;
}

/* Full thunk: saves all 16 ymm (256-bit). Used for any run with a vector op. */
void *avxemu_build_thunk(const tramp_insn *insns, int n, uint64_t resume) {
    return build_thunk_t(insns, n, resume, avxemu_tt_start, avxemu_tt_record,
                         avxemu_tt_dispatchptr, avxemu_tt_resumeptr,
                         (void *)avxemu_tramp_dispatch);
}

/* GPR-only thunk: saves only xmm0-15 (128-bit). SAFE only for runs whose every
 * instruction is a GPR-domain op (is_bmi: BMI1/2, LZCNT, TZCNT, MOVBE) — those
 * touch no vector register, so the emulation neither reads nor writes ymm and the
 * upper 128 bits are left untouched by the SSE-only C path. Halves the per-run
 * vector spill, the dominant trampoline cost. */
static void *build_thunk_gpr(const tramp_insn *insns, int n, uint64_t resume) {
    return build_thunk_t(insns, n, resume, avxemu_ttg_start, avxemu_ttg_record,
                         avxemu_ttg_dispatchptr, avxemu_ttg_resumeptr,
                         (void *)avxemu_tramp_dispatch);
}

/* BMI reg-only thunk: saves GPRs+flags only, no vector save. Dispatched through
 * the xmm-clean avxemu_tramp_dispatch_bmi, so the program's vector state is
 * untouched and needs no preservation. */
static void *build_thunk_bmi(const tramp_insn *insns, int n, uint64_t resume) {
    return build_thunk_t(insns, n, resume, avxemu_tt2_start, avxemu_tt2_record,
                         avxemu_tt2_dispatchptr, avxemu_tt2_resumeptr,
                         (void *)avxemu_tramp_dispatch_bmi);
}

/* True iff every instruction in the run is a GPR-domain op (no vector reg). */
static int run_is_gpr_only(const tramp_insn *insns, int n) {
    for (int i = 0; i < n; i++) if (!insns[i].dec.is_bmi) return 0;
    return n > 0;
}
/* Stricter: every insn is a register-operand BMI op (no memory, no segment) -> the
 * emulation touches only GPRs, so the no-vector-save tt2 thunk is safe. */
static int run_is_regonly_bmi(const tramp_insn *insns, int n) {
    for (int i = 0; i < n; i++) {
        const decoded *d = &insns[i].dec;
        if (!d->is_bmi || d->a_src == OPND_MEM || d->b_src == OPND_MEM
            || d->dst_kind == DST_MEM || d->seg != 0) return 0;
    }
    return n > 0;
}

/* ----------------------------------------------------------------------------
 * Eager installer: at load, rewrite each run of faulting instructions to a jmp
 * into a thunk, so it never traps. Runs single-threaded in the constructor,
 * before the patched code can execute.
 * -------------------------------------------------------------------------- */

static int g_lack_avx2 = 1, g_lack_fma = 1, g_lack_bmi = 1, g_lack_f16c = 0;
static int g_force_full = 0;   /* AVXEMU_FULLTHUNK: force the register-saving thunk (correctness probe) */

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
/* Does the instruction #UD on this CPU (and so currently go through SIGILL)? */
static int tramp_faults(const decoded *d) {
    if (d->is_bmi) return g_lack_bmi;
    if (d->op >= VFMADD132 && d->op <= VFNMSUB231) return g_lack_fma;
    if (d->op == VCVTPH2PS) return g_lack_f16c;
    if (avx2_only_op(d->op)) return g_lack_avx2;
    return d->wide && g_lack_avx2;        /* base SSE integer op: only 256-bit faults */
}

static void detect_features(void) {
    g_force_full = getenv("AVXEMU_FULLTHUNK") != 0;
    if (getenv("AVXEMU_FORCETRAMP")) {    /* dev: treat everything emulatable as faulting */
        g_lack_avx2 = g_lack_fma = g_lack_bmi = g_lack_f16c = 1; return;
    }
    unsigned a, b, c, d;
    if (__get_cpuid(1, &a, &b, &c, &d)) { g_lack_fma = !(c & (1u<<12)); g_lack_f16c = !(c & (1u<<29)); }
    unsigned a7 = 0, b7 = 0, c7 = 0, d7 = 0;
    __cpuid_count(7, 0, a7, b7, c7, d7);
    g_lack_avx2 = !(b7 & (1u<<5));
    int bmi1 = (b7 & (1u<<3)) != 0, bmi2 = (b7 & (1u<<8)) != 0;
    unsigned a8, b8, c8, d8; int lz = __get_cpuid(0x80000001u, &a8, &b8, &c8, &d8) && (c8 & (1u<<5));
    g_lack_bmi = !(bmi1 && bmi2 && lz);   /* conservative: any BMI-ish missing -> emulate the family */
}

#define MAXRUN 64

/* ============================================================================
 * Milestone B: register-resident native-SSE codegen.
 *
 * For a run whose every instruction is one of a small set of dominant AVX2
 * vector ops, instead of routing through avxemu_tramp_dispatch (a C call that
 * loops avxemu_emulate per instruction), we EMIT a native straight-line SSE
 * function `block(const run_record *r, avxemu_regfile *rf)` (same signature the
 * tt template calls: rdi=run_record (unused here), rsi=&rf). The block reads
 * each op's source operands out of the in-memory regfile (rf), computes the op
 * with native SSE on the two 128-bit halves, and writes the result back into
 * rf. The tt template's spill/reload/resume is unchanged — only the dispatch
 * pointer is swapped from avxemu_tramp_dispatch to this emitted block.
 *
 * CORRECTNESS MODEL (write-through): every instruction's result is stored back
 * to its rf ymm slot immediately, so the in-memory rf is the canonical register
 * file and is up to date at every instruction boundary. Data dependencies
 * within the run therefore resolve through rf automatically. As a register-
 * residency optimisation for the dominant straight-line dependency chain, the
 * LOW 128 bits of the immediately-preceding destination are kept live in an xmm
 * (NX_CACHE) and reused when the next instruction's source slot matches — this
 * is a pure copy of what was just written to rf, so it can never be stale. The
 * HIGH 128 bits are always reloaded from rf (cheap, and sidesteps any stale-
 * upper-half pitfall). The block clobbers only caller-saved GPRs (rax/rcx) and
 * xmm; the tt template reloads ALL of rf into the real registers after the call,
 * so any clobber is irrelevant. rsi (rf) and rbx/rsp/rbp/r12-15 are preserved.
 *
 * This file stays VEX-CLEAN: we only WRITE machine-code bytes (data) for the SSE
 * (and VEX-encoded source, never compiled) instructions; the compiled C emits no
 * VEX. The emitted block uses only SSE2/SSE4.1 (pand/por/pxor/psubb/pcmpeqb/
 * pcmpgtb/pshufd/movd/pmovzxbw/psrldq), all present on the target Ivy Bridge.
 *
 * Op set: VPBROADCASTD, VPAND, VPOR, VPXOR, VPSUBB, VPCMPEQB, VPCMPGTB,
 * VPMOVZXBW (each in register and memory-operand form). Any op outside the set,
 * a segment override (seg!=0), or a memory destination -> decline the WHOLE run
 * (avxemu_build_thunk_native returns NULL; caller falls back to C dispatch).
 * Declining is never a correctness risk.
 * ========================================================================== */
#define RF_GPR_OFF 512   /* byte offset of gpr[0] within avxemu_regfile (16*32) */
/* xmm register roles inside the emitted block (NX_ZERO is a pinned zero reg) */
enum { NX_A_LO = 0, NX_A_HI = 1, NX_B_LO = 2, NX_B_HI = 3, NX_TMP = 4,
       NX_CACHE = 6, NX_ZERO = 15 };
#define NB_RSI 6   /* rf base pointer register (arg2) */
#define NB_RAX 0   /* EA accumulator */
#define NB_RCX 1   /* index scratch */

static void nb(uint8_t **p, uint8_t b){ *(*p)++ = b; }
static void nb32(uint8_t **p, uint32_t v){ memcpy(*p, &v, 4); *p += 4; }
static void nb64(uint8_t **p, uint64_t v){ memcpy(*p, &v, 8); *p += 8; }

/* SSE op  xmm[reg] , xmm[rm]   (ModRM mod=11). pre = mandatory/legacy prefix
 * (0x66/0xF3/0), esc38 = use the 0F38 escape map. */
static void sse_rr(uint8_t **p, uint8_t pre, int esc38, uint8_t op,
                   int reg, int rm, int has_imm, uint8_t imm){
    if (pre) nb(p, pre);
    uint8_t rex = (uint8_t)(((reg >= 8) ? 4 : 0) | ((rm >= 8) ? 1 : 0));
    if (rex) nb(p, (uint8_t)(0x40 | rex));
    nb(p, 0x0F); if (esc38) nb(p, 0x38); nb(p, op);
    nb(p, (uint8_t)(0xC0 | ((reg & 7) << 3) | (rm & 7)));
    if (has_imm) nb(p, imm);
}
/* SSE op  xmm[reg] , [base + disp32]   (ModRM mod=10). base must not be rsp/r12
 * (would need a SIB) — all callers use rsi or rax/rcx, never reg 4. */
static void sse_rm(uint8_t **p, uint8_t pre, int esc38, uint8_t op,
                   int reg, int base, int32_t disp, int has_imm, uint8_t imm){
    if (pre) nb(p, pre);
    uint8_t rex = (uint8_t)(((reg >= 8) ? 4 : 0) | ((base >= 8) ? 1 : 0));
    if (rex) nb(p, (uint8_t)(0x40 | rex));
    nb(p, 0x0F); if (esc38) nb(p, 0x38); nb(p, op);
    nb(p, (uint8_t)(0x80 | ((reg & 7) << 3) | (base & 7)));
    nb32(p, (uint32_t)disp);
    if (has_imm) nb(p, imm);
}
static void nb_movdqu_load (uint8_t **p, int xmm, int base, int32_t d){ sse_rm(p, 0xF3, 0, 0x6F, xmm, base, d, 0, 0); }
static void nb_movdqu_store(uint8_t **p, int xmm, int base, int32_t d){ sse_rm(p, 0xF3, 0, 0x7F, xmm, base, d, 0, 0); }
static void nb_movdqa      (uint8_t **p, int dst, int src){ sse_rr(p, 0x66, 0, 0x6F, dst, src, 0, 0); }

/* GPR EA helpers. All operate at 64-bit (REX.W). The memory base for the loads
 * is always rsi (rf), encoded with mod=10 disp32. */
static void nb_movabs_rax (uint8_t **p, uint64_t v){ nb(p, 0x48); nb(p, 0xB8); nb64(p, v); }
static void nb_add_rax_mem(uint8_t **p, int32_t d){ nb(p, 0x48); nb(p, 0x03); nb(p, (uint8_t)(0x80 | (0 << 3) | NB_RSI)); nb32(p, (uint32_t)d); }
static void nb_mov_rcx_mem(uint8_t **p, int32_t d){ nb(p, 0x48); nb(p, 0x8B); nb(p, (uint8_t)(0x80 | (1 << 3) | NB_RSI)); nb32(p, (uint32_t)d); }
static void nb_shl_rcx    (uint8_t **p, uint8_t imm){ nb(p, 0x48); nb(p, 0xC1); nb(p, 0xE1); nb(p, imm); }
static void nb_add_rax_rcx(uint8_t **p){ nb(p, 0x48); nb(p, 0x01); nb(p, 0xC8); }

/* Compute the memory operand's effective address into rax, exactly mirroring
 * ea_rf(): EA = disp + (rip_rel ? insn_addr+len : 0) + gpr[base] + gpr[index]*scale.
 * Both the disp and the rip-relative term are known at emit time (insn_addr is
 * the instruction's runtime address, recorded in the run_record), so they fold
 * into a single 64-bit immediate. base/index are read live from rf->gpr. */
static void emit_ea(uint8_t **p, const decoded *d, uint64_t addr){
    uint64_t rip_next = addr + d->len;
    int64_t c = (int64_t)d->disp + (d->rip_rel ? (int64_t)rip_next : 0);
    nb_movabs_rax(p, (uint64_t)c);
    if (d->base != OPND_NONE)  nb_add_rax_mem(p, RF_GPR_OFF + (int)d->base * 8);
    if (d->index != OPND_NONE){
        nb_mov_rcx_mem(p, RF_GPR_OFF + (int)d->index * 8);
        uint8_t sh = (d->scale == 8) ? 3 : (d->scale == 4) ? 2 : (d->scale == 2) ? 1 : 0;
        if (sh) nb_shl_rcx(p, sh);
        nb_add_rax_rcx(p);
    }
}

static int native_op_supported(vex_op op){
    switch (op){
    case VPBROADCASTD: case VPBROADCASTW: case VPAND: case VPOR: case VPXOR:
    case VPSUBB: case VPCMPEQB: case VPCMPGTB: case VPMOVZXBW: return 1;
    default: return 0;
    }
}
static int native_is_binop(vex_op op){
    switch (op){
    case VPAND: case VPOR: case VPXOR: case VPSUBB: case VPCMPEQB: case VPCMPGTB: return 1;
    default: return 0;
    }
}
static int native_insn_supported(const decoded *d){
    if (!native_op_supported(d->op)) return 0;
    if (d->seg != 0) return 0;                 /* gs/fs: not supported -> decline run */
    if (d->dst_kind != DST_YMM) return 0;      /* memory dest not implemented -> decline */
    if (d->dst < 0 || d->dst > 15) return 0;
    if (native_is_binop(d->op)){               /* src1 (VEX.vvvv) is always a register */
        if (d->a_src < 0 || d->a_src > 15) return 0;
    }
    if (d->b_src == OPND_MEM){
        if (d->base  != OPND_NONE && (d->base  < 0 || d->base  > 15)) return 0;
        if (d->index != OPND_NONE && (d->index < 0 || d->index > 15)) return 0;
    } else if (d->b_src < 0 || d->b_src > 15) return 0;
    return 1;
}

/* Load register-slot's low (and, if wide, high) 128 bits into lo_reg/hi_reg.
 * Low half reuses NX_CACHE when the slot is the previous destination. */
static void load_slot(uint8_t **p, int slot, int lo_reg, int hi_reg, int wide, int prev_dst){
    if (slot == prev_dst) nb_movdqa(p, lo_reg, NX_CACHE);
    else                  nb_movdqu_load(p, lo_reg, NB_RSI, slot * 32);
    if (wide)             nb_movdqu_load(p, hi_reg, NB_RSI, slot * 32 + 16);
}
/* Store result to dst slot: low always; high if wide, else zero the upper 128
 * (VEX.128 semantics — matches avxemu_emulate's memset of the upper half). */
static void store_dst(uint8_t **p, int slot, int lo_reg, int hi_reg, int wide){
    nb_movdqu_store(p, lo_reg, NB_RSI, slot * 32);
    if (wide) nb_movdqu_store(p, hi_reg, NB_RSI, slot * 32 + 16);
    else      nb_movdqu_store(p, NX_ZERO, NB_RSI, slot * 32 + 16);
}

/* Emit one instruction. EA (if any) is computed into rax first. Returns 1 on
 * success, 0 if (defensively) unhandled — caller then declines the run. */
static int emit_one(uint8_t **p, const decoded *d, uint64_t addr, int *prev_dst){
    int wide = d->wide;
    int D = d->dst;
    int mem = (d->b_src == OPND_MEM);
    if (mem) emit_ea(p, d, addr);     /* rax = &memory operand */

    switch (d->op){
    case VPAND: case VPOR: case VPXOR: case VPSUBB: case VPCMPEQB: case VPCMPGTB: {
        uint8_t opc = (d->op == VPAND)    ? 0xDB : (d->op == VPOR)     ? 0xEB :
                      (d->op == VPXOR)    ? 0xEF : (d->op == VPSUBB)   ? 0xF8 :
                      (d->op == VPCMPEQB) ? 0x74 : /* VPCMPGTB */        0x64;
        load_slot(p, d->a_src, NX_A_LO, NX_A_HI, wide, *prev_dst);
        if (mem){
            nb_movdqu_load(p, NX_B_LO, NB_RAX, 0);
            if (wide) nb_movdqu_load(p, NX_B_HI, NB_RAX, 16);
        } else {
            load_slot(p, d->b_src, NX_B_LO, NX_B_HI, wide, *prev_dst);
        }
        sse_rr(p, 0x66, 0, opc, NX_A_LO, NX_B_LO, 0, 0);          /* res.lo = a.lo OP b.lo */
        if (wide) sse_rr(p, 0x66, 0, opc, NX_A_HI, NX_B_HI, 0, 0);
        store_dst(p, D, NX_A_LO, NX_A_HI, wide);
        nb_movdqa(p, NX_CACHE, NX_A_LO); *prev_dst = D;
        break;
    }
    case VPBROADCASTD: {
        /* load the source dword into the low dword, then pshufd $0 -> all lanes */
        if (mem) sse_rm(p, 0x66, 0, 0x6E, NX_A_LO, NB_RAX, 0, 0, 0);          /* movd A_LO,[rax]      */
        else     sse_rm(p, 0x66, 0, 0x6E, NX_A_LO, NB_RSI, d->b_src * 32, 0, 0);/* movd A_LO,[rsi+b*32] */
        sse_rr(p, 0x66, 0, 0x70, NX_A_LO, NX_A_LO, 1, 0x00);                  /* pshufd A_LO,A_LO,0   */
        if (wide) nb_movdqa(p, NX_A_HI, NX_A_LO);
        store_dst(p, D, NX_A_LO, NX_A_HI, wide);
        nb_movdqa(p, NX_CACHE, NX_A_LO); *prev_dst = D;
        break;
    }
    case VPBROADCASTW: {
        /* broadcast word 0 of the source to all 8 word lanes (per 128). For a
         * memory source, read EXACTLY 2 bytes (pinsrw) so we never over-read past
         * the word vpbroadcastw would touch; for a register slot, movd the low
         * dword (its low word is word 0). Then pshuflw $0 replicates word 0 into
         * the low 4 words, and pshufd $0 replicates the low qword across all 4
         * dwords -> all 8 words = word 0. Matches vec_exec's 16x uint16 store. */
        if (mem) sse_rm(p, 0x66, 0, 0xC4, NX_A_LO, NB_RAX, 0, 1, 0x00);        /* pinsrw A_LO,[rax],0  */
        else     sse_rm(p, 0x66, 0, 0x6E, NX_A_LO, NB_RSI, d->b_src * 32, 0, 0);/* movd  A_LO,[rsi+b*32] */
        sse_rr(p, 0xF2, 0, 0x70, NX_A_LO, NX_A_LO, 1, 0x00);                  /* pshuflw A_LO,A_LO,0  */
        sse_rr(p, 0x66, 0, 0x70, NX_A_LO, NX_A_LO, 1, 0x00);                  /* pshufd  A_LO,A_LO,0  */
        if (wide) nb_movdqa(p, NX_A_HI, NX_A_LO);
        store_dst(p, D, NX_A_LO, NX_A_HI, wide);
        nb_movdqa(p, NX_CACHE, NX_A_LO); *prev_dst = D;
        break;
    }
    case VPMOVZXBW: {
        /* zero-extend bytes->words. Low lane <- src bytes 0..7; (wide) high lane
         * <- src bytes 8..15. Matches vec_exec's out[i]=src[i] across 16 words. */
        if (mem){
            sse_rm(p, 0x66, 1, 0x30, NX_A_LO, NB_RAX, 0, 0, 0);              /* pmovzxbw A_LO,[rax]   */
            if (wide) sse_rm(p, 0x66, 1, 0x30, NX_A_HI, NB_RAX, 8, 0, 0);    /* pmovzxbw A_HI,[rax+8] */
        } else {
            if (d->b_src == *prev_dst) nb_movdqa(p, NX_B_LO, NX_CACHE);
            else nb_movdqu_load(p, NX_B_LO, NB_RSI, d->b_src * 32);
            sse_rr(p, 0x66, 1, 0x30, NX_A_LO, NX_B_LO, 0, 0);               /* pmovzxbw A_LO,B_LO    */
            if (wide){
                nb_movdqa(p, NX_TMP, NX_B_LO);
                sse_rr(p, 0x66, 0, 0x73, 3, NX_TMP, 1, 8);                  /* psrldq TMP,8 (/3)     */
                sse_rr(p, 0x66, 1, 0x30, NX_A_HI, NX_TMP, 0, 0);           /* pmovzxbw A_HI,TMP     */
            }
        }
        store_dst(p, D, NX_A_LO, NX_A_HI, wide);
        nb_movdqa(p, NX_CACHE, NX_A_LO); *prev_dst = D;
        break;
    }
    default: return 0;
    }
    return 1;
}

/* AVXEMU_NATIVE gate (default ON; =0 forces the legacy C-dispatch thunks). */
static int g_native = -1;
static int native_enabled(void){
    if (g_native < 0){ const char *e = getenv("AVXEMU_NATIVE"); g_native = (e && e[0] == '0') ? 0 : 1; }
    return g_native;
}

/* ---- AVXEMU_NATIVE_STATS: build-time run-composition tally (env-gated). Counts
 * runs the native emitter would accept vs decline, run-length single vs multi,
 * and for declined runs the first unsupported (blocking) op. Printed at end of
 * install_trampolines. Pure diagnostic; no effect on what gets emitted. */
static int g_nstats = -1;
static int nstats_enabled(void){
    if (g_nstats < 0){ const char *e = getenv("AVXEMU_NATIVE_STATS"); g_nstats = (e && e[0] && e[0] != '0') ? 1 : 0; }
    return g_nstats;
}
static uint64_t g_ns_accept, g_ns_decline, g_ns_single, g_ns_multi;
static uint64_t g_ns_block_op[VEX_OP_COUNT];
static void nstats_tally(const tramp_insn *ri, int rn){
    if (rn == 1) g_ns_single++; else g_ns_multi++;
    int blk = -1;
    for (int k = 0; k < rn; k++) if (!native_insn_supported(&ri[k].dec)){ blk = ri[k].dec.op; break; }
    if (blk < 0) g_ns_accept++;
    else { g_ns_decline++; g_ns_block_op[blk]++; }
}

/* Emit the native SSE `block` for a run into the RWX pool. Returns the block
 * entry, or NULL if any instruction is unsupported or the pool is exhausted.
 * Bypasses the AVXEMU_NATIVE gate (raw emitter; the gate is in the thunk
 * builder) so the differential test can always exercise it. */
void *avxemu_emit_native_block(const tramp_insn *insns, int n){
    if (n <= 0) return 0;
    for (int i = 0; i < n; i++) if (!native_insn_supported(&insns[i].dec)) return 0;
    uint8_t buf[MAXRUN * 160];
    uint8_t *p = buf;
    sse_rr(&p, 0x66, 0, 0xEF, NX_ZERO, NX_ZERO, 0, 0);   /* pxor xmm15,xmm15 (zero reg) */
    int prev_dst = -1;
    for (int i = 0; i < n; i++){
        if (!emit_one(&p, &insns[i].dec, insns[i].addr, &prev_dst)) return 0;
        if ((size_t)(p - buf) > sizeof buf - 128) return 0;   /* overflow guard */
    }
    nb(&p, 0xC3);   /* ret */
    size_t sz = (size_t)(p - buf);
    uint8_t *blk = avxemu_pool_alloc(sz);
    if (!blk) return 0;
    memcpy(blk, buf, sz);
    return blk;
}

/* Native-SSE thunk: a tt-template thunk whose dispatch pointer is the emitted
 * native block instead of avxemu_tramp_dispatch. The spill/reload/resume is the
 * unchanged tt template. Returns NULL (caller falls back) if disabled, any op is
 * unsupported, or the pool is exhausted. */
void *avxemu_build_thunk_native(const tramp_insn *insns, int n, uint64_t resume){
    if (!native_enabled()) return 0;
    void *block = avxemu_emit_native_block(insns, n);
    if (!block) return 0;
    return build_thunk_t(insns, n, resume, avxemu_tt_start, avxemu_tt_record,
                         avxemu_tt_dispatchptr, avxemu_tt_resumeptr, block);
}

/* ============================================================================
 * Milestone B (Task 2): scalar-GPR native codegen for the trampoline.
 *
 * Trampolined LZCNT currently runs through the tt2 (GPR+flags spill) thunk ->
 * avxemu_tramp_dispatch_bmi -> per-instruction C (bmi_exec). This emits a native
 * straight-line block(run_record* rdi, avxemu_regfile* rf=rsi) that operates on
 * the SPILLED rf->gpr[] slots (rsi+512+reg*8) and rf->rflags slot (rsi+640) — the
 * exact same memory image the tt2 thunk saved — then returns. The tt2 thunk
 * reloads every GPR + popf's rflags from those slots afterward, so the block may
 * clobber any caller-saved GPR; it only must preserve rsi (its base) and rsp.
 *
 * Host is no-BMI/no-LZCNT Ivy Bridge: the lowering uses ONLY base-ISA GPR ops
 * (bsr + fixups, NEVER lzcnt — which would #UD). This file stays VEX-clean: we
 * only WRITE machine-code bytes (data); the compiled C emits no VEX and (for this
 * GPR-domain block) no SSE either, so it is safe under the no-vector-save tt2.
 *
 * LZCNT semantics, bit-exact to bmi_exec/set_flags:
 *   result = (src==0) ? opsize : (opsize-1 - bsr(src))
 *   CF = (src==0);  ZF = (result==0);  SF = 0;  OF = 0;  PF/AF preserved.
 *   opsize 32 zero-extends the dst slot's full 64 bits.
 * Only the owned flag bits (CF|ZF|SF|OF) of the rf->rflags slot are rewritten
 * (read-modify-write with mask ~0x8C1), preserving PF/AF and every other bit.
 * ========================================================================== */
#define RF_FLAGS_OFF 640   /* byte offset of rflags within avxemu_regfile */
/* fixed scratch CPU regs for the block (all caller-saved & reloaded by tt2; none
 * is rsi=6, the rf base): S=rax, D=rcx, T1=rdx, T2=r8, F=r9. The decoded dst/src
 * are SLOT indices (memory), independent of these. */
enum { GB_S = 0, GB_D = 1, GB_T1 = 2, GB_T2 = 8, GB_F = 9 };

static void gb_rex(uint8_t **p, int w, int reg, int rm){
    uint8_t r = (uint8_t)((w?8:0) | ((reg>=8)?4:0) | ((rm>=8)?1:0));
    if (r) nb(p, (uint8_t)(0x40 | r));
}
/* mov reg, [rsi+disp32]   (w=1: 64-bit; w=0: 32-bit, zero-extends to 64) */
static void gb_load(uint8_t **p, int reg, int32_t disp, int w){
    gb_rex(p, w, reg, NB_RSI);
    nb(p, 0x8B); nb(p, (uint8_t)(0x80 | ((reg&7)<<3) | NB_RSI)); nb32(p, (uint32_t)disp);
}
/* mov [rsi+disp32], reg   (always 64-bit: writes the full slot) */
static void gb_store(uint8_t **p, int reg, int32_t disp){
    gb_rex(p, 1, reg, NB_RSI);
    nb(p, 0x89); nb(p, (uint8_t)(0x80 | ((reg&7)<<3) | NB_RSI)); nb32(p, (uint32_t)disp);
}
static void gb_alu_rr(uint8_t **p, int w, uint8_t opc, int reg, int rm){
    gb_rex(p, w, reg, rm); nb(p, opc); nb(p, (uint8_t)(0xC0 | ((reg&7)<<3) | (rm&7)));
}
static void gb_alu0f_rr(uint8_t **p, int w, uint8_t opc, int reg, int rm){
    gb_rex(p, w, reg, rm); nb(p, 0x0F); nb(p, opc); nb(p, (uint8_t)(0xC0 | ((reg&7)<<3) | (rm&7)));
}
static void gb_xor  (uint8_t **p,int w,int d,int s){ gb_alu_rr(p,w,0x31,s,d); }  /* d ^= s   */
static void gb_or   (uint8_t **p,int w,int d,int s){ gb_alu_rr(p,w,0x09,s,d); }  /* d |= s   */
static void gb_sub  (uint8_t **p,int w,int d,int s){ gb_alu_rr(p,w,0x29,s,d); }  /* d -= s   */
static void gb_test (uint8_t **p,int w,int a,int b){ gb_alu_rr(p,w,0x85,a,b); }  /* test a,b */
static void gb_bsr  (uint8_t **p,int w,int d,int s){ gb_alu0f_rr(p,w,0xBD,d,s); }
static void gb_cmovz(uint8_t **p,int w,int d,int s){ gb_alu0f_rr(p,w,0x44,d,s); }
/* setz r8l. Force a REX so r8..r15 (and r4..r7) select the proper low byte. */
static void gb_setz (uint8_t **p,int r){ nb(p,(uint8_t)(0x40|((r>=8)?1:0))); nb(p,0x0F); nb(p,0x94); nb(p,(uint8_t)(0xC0|(r&7))); }
static void gb_shl_imm(uint8_t **p,int w,int r,uint8_t i){ gb_rex(p,w,0,r); nb(p,0xC1); nb(p,(uint8_t)(0xC0|(4<<3)|(r&7))); nb(p,i); }
static void gb_mov_imm32(uint8_t **p,int r,uint32_t imm){ if(r>=8) nb(p,0x41); nb(p,(uint8_t)(0xB8|(r&7))); nb32(p,imm); } /* zero-extends to 64 */
static void gb_and_imm32(uint8_t **p,int w,int r,uint32_t imm){ gb_rex(p,w,0,r); nb(p,0x81); nb(p,(uint8_t)(0xC0|(4<<3)|(r&7))); nb32(p,imm); } /* sign-extends imm32 */

/* Emit one LZCNT into the scalar block. Mirrors reloc.c's LZCNT lowering for the
 * value+flag arithmetic, but reads src and writes dst/flags via the rf MEMORY
 * slots (not live registers), and patches the rf->rflags slot instead of a saved
 * pushf image. dst==src is safe: src is loaded into a register first, and the dst
 * slot is written only at the very end. */
static int gb_emit_lzcnt(uint8_t **p, const decoded *d){
    const int w = (d->opsize == 64);
    const uint32_t n = d->opsize;
    const uint32_t CLR = 0xFFFFF73Eu;   /* ~(CF|ZF|SF|OF) = ~0x8C1, sign-extended to 64 in AND */
    const int S = GB_S, D = GB_D, T1 = GB_T1, T2 = GB_T2, F = GB_F;

    gb_load(p, S, RF_GPR_OFF + d->a_src * 8, w);   /* S = rf->gpr[src] (masked to opsize) */
    gb_xor (p, 1, T1, T1);                          /* T1 = 0 */
    gb_test(p, w, S, S);                            /* ZF = (src==0)  [src not clobbered below] */
    gb_setz(p, T1);                                 /* T1 = (src==0) -> CF */
    gb_bsr (p, w, D, S);                            /* D = bsr(src)  (undef if src==0) */
    gb_mov_imm32(p, T2, n - 1);                     /* T2 = opsize-1 */
    gb_sub (p, 1, T2, D);                           /* T2 = (opsize-1) - bsr = lzcnt for src!=0 */
    gb_test(p, 1, T1, T1);                          /* ZF = (T1==0) = (src!=0) */
    gb_mov_imm32(p, D, n);                          /* D = opsize (zero-input result) */
    gb_cmovz(p, 1, D, T2);                          /* src!=0 -> D = T2 */
    /* ZF = (result==0) -> bit6; SF=0, OF=0 */
    gb_xor (p, 1, T2, T2);
    gb_test(p, w, D, D);
    gb_setz(p, T2);
    gb_shl_imm(p, 1, T2, 6);
    gb_or  (p, 1, T1, T2);                          /* owned = CF | ZF */
    /* patch rf->rflags: clear owned bits, OR in computed, preserve PF/AF/rest */
    gb_load(p, F, RF_FLAGS_OFF, 1);
    gb_and_imm32(p, 1, F, CLR);
    gb_or  (p, 1, F, T1);
    gb_store(p, F, RF_FLAGS_OFF);
    gb_store(p, D, RF_GPR_OFF + d->dst * 8);        /* result (upper 32 zero for opsize 32) */
    return 1;
}

/* Is this insn a scalar-GPR op the native scalar block supports? (LZCNT only for
 * now; register operands, no memory/segment, opsize 32/64.) */
static int bmi_native_insn_supported(const decoded *d){
    if (!d->is_bmi || d->op != BMI_LZCNT) return 0;
    if (d->dst_kind != DST_GPR) return 0;
    if (!(d->opsize == 32 || d->opsize == 64)) return 0;
    if (d->seg != 0) return 0;
    if (d->a_src == OPND_MEM || d->b_src == OPND_MEM || d->dst_kind == DST_MEM) return 0;
    if (d->dst < 0 || d->dst > 15) return 0;
    if (d->a_src < 0 || d->a_src > 15) return 0;
    if (d->bmi_dst2 != OPND_NONE || d->bmi_s1_rdx) return 0;
    return 1;
}

/* Emit the native scalar-GPR `block` for a run into the RWX pool. Returns the
 * block entry, or NULL if any instruction is unsupported or the pool is
 * exhausted. Bypasses the AVXEMU_NATIVE gate (the gate is in the thunk builder)
 * so the differential test can always exercise it. */
void *avxemu_emit_native_block_bmi(const tramp_insn *insns, int n){
    if (n <= 0) return 0;
    for (int i = 0; i < n; i++) if (!bmi_native_insn_supported(&insns[i].dec)) return 0;
    uint8_t buf[MAXRUN * 160];
    uint8_t *p = buf;
    for (int i = 0; i < n; i++){
        if (!gb_emit_lzcnt(&p, &insns[i].dec)) return 0;
        if ((size_t)(p - buf) > sizeof buf - 128) return 0;   /* overflow guard */
    }
    nb(&p, 0xC3);   /* ret */
    size_t sz = (size_t)(p - buf);
    uint8_t *blk = avxemu_pool_alloc(sz);
    if (!blk) return 0;
    memcpy(blk, buf, sz);
    return blk;
}

/* True iff every instruction in the run is a scalar-GPR op the native block
 * supports (currently: register-operand LZCNT). */
static int run_is_native_bmi(const tramp_insn *insns, int n){
    for (int i = 0; i < n; i++) if (!bmi_native_insn_supported(&insns[i].dec)) return 0;
    return n > 0;
}

/* Native scalar-GPR thunk: a tt2-template thunk (GPR+flags spill, NO vector save
 * — correct since LZCNT touches no vector reg) whose dispatch pointer is the
 * emitted scalar block. Returns NULL (caller falls back) if disabled, any op is
 * unsupported, or the pool is exhausted. */
void *avxemu_build_thunk_native_bmi(const tramp_insn *insns, int n, uint64_t resume){
    if (!native_enabled()) return 0;
    void *block = avxemu_emit_native_block_bmi(insns, n);
    if (!block) return 0;
    return build_thunk_t(insns, n, resume, avxemu_tt2_start, avxemu_tt2_record,
                         avxemu_tt2_dispatchptr, avxemu_tt2_resumeptr, block);
}

/* Scan one cleanly-decoding function and trampoline its faulting runs. Returns
 * the number of jmps written. Patches go to the live (already-writable) text. */
/* Place the trampoline for one gathered run of faulting instructions: pick the
 * first start whose 5-byte jmp can't corrupt a branch target, build a thunk for
 * [s,ni), and overwrite that start with `jmp thunk`. Returns 1 if patched. */
static long emit_run(uint8_t *text, size_t fstart, size_t fend,
                     const tramp_insn *insns, const size_t *offs, int ni,
                     size_t re, const uint8_t *tgt, int has_indirect) {
    int s = 0;
    while (s < ni) {
        size_t soff = offs[s]; int slen = insns[s].dec.len;
        if (re - soff < 5) { s = ni; break; }            /* can't fit a 5-byte jmp -> rest traps */
        if (slen >= 5) break;                            /* jmp lands inside insn[s]: always safe */
        size_t nb = soff + 4;                            /* slen==4: next insn start is in the jmp */
        if (!has_indirect && (nb >= fend || !tgt[nb - fstart])) break;
        s++;                                              /* unsafe 4-byte start -> it traps; try next */
    }
    if (s < ni && re - offs[s] >= 5) {
        const tramp_insn *ri = insns + s; int rn = ni - s;
        if (g_nstats) nstats_tally(ri, rn);   /* AVXEMU_NATIVE_STATS diagnostic */
        uint64_t res = (uint64_t)(text + re);
        /* Milestone B: try the register-resident native-SSE thunk first (gated by
         * AVXEMU_NATIVE, default ON). On NULL — op outside the supported set, a
         * segment/mem-dest operand, pool exhausted, or gate off — fall back to the
         * existing C-dispatch thunk selection (full/gpr/bmi). No correctness risk. */
        void *thunk = g_force_full ? 0 : avxemu_build_thunk_native(ri, rn, res);
        /* Task 2: scalar-GPR native (register-only LZCNT) via the tt2 thunk. The
         * vector native attempt above declines BMI ops, so this runs next. NULL
         * (gate off, op outside the scalar set, or pool exhausted) -> existing C
         * thunk selection. No correctness risk. */
        if (!thunk && !g_force_full && run_is_native_bmi(ri, rn))
            thunk = avxemu_build_thunk_native_bmi(ri, rn, res);
        if (!thunk)
            thunk = g_force_full              ? avxemu_build_thunk(ri, rn, res)
                  : run_is_regonly_bmi(ri, rn) ? build_thunk_bmi(ri, rn, res)
                  : run_is_gpr_only(ri, rn)    ? build_thunk_gpr(ri, rn, res)
                  :                              avxemu_build_thunk(ri, rn, res);
        if (thunk) {
            uint8_t *site = text + offs[s];
            int64_t rel = (int64_t)((uint8_t *)thunk - (site + 5));
            if (rel >= INT32_MIN && rel <= INT32_MAX) {
                site[0] = 0xE9; int32_t r32 = (int32_t)rel; memcpy(site + 1, &r32, 4);
                return 1;
            }
        }
    }
    return 0;
}

/* Gather the maximal run of physically-consecutive faulting instructions at q
 * (each must also be reachable code when `code` is given), fill insns/offs, and
 * return the run-end offset. Faulting ops never branch, so the run always falls
 * through to real code at the returned offset. */
static size_t gather_run(uint8_t *text, size_t fstart, size_t fend, size_t q,
                         const uint8_t *code, tramp_insn *insns, size_t *offs, int *pni) {
    int ni = 0; size_t p = q;
    while (p < fend && ni < MAXRUN && (!code || code[p - fstart])) {
        int z2, oo; int l2 = x86_len(text + p, text + fend, &z2, &oo);
        if (l2 <= 0) break;
        decoded d2; int dl2 = decode(text + p, &d2);
        if (!(dl2 > 0 && d2.op && tramp_faults(&d2))) break;
        offs[ni] = p; insns[ni].addr = (uint64_t)(text + p); insns[ni].dec = d2;
        ni++; p += l2;
    }
    *pni = ni;
    return p;
}

/* ---- bounded jump-table resolution (fault-storm fix) ------------------------
 * The dominant repeat-faulting sites live in functions whose only indirect jmps
 * are LLVM switch dispatches:
 *     cmp  rI, imm          ; unsigned bound: valid index 0..imm
 *     ja   <default>        ; forward, past the dispatch
 *     lea  rB, [rip+d32]    ; table base (rel32-entry table, usually inline)
 *     movsxd rD, [rB+rI*4]  ; load rel32 entry
 *     add  rD, rB           ; absolute target
 *     jmp  *rD
 * That target set is STATICALLY ENUMERABLE: entries e_i at [tbl, tbl+4*(imm+1)),
 * targets tbl+e_i. resolve_jump_table() matches the pattern against the last few
 * decoded instructions ending at the `jmp *rD` and, on an EXACT match, marks
 * every in-function target in tgt[] and records the inline table bytes as data
 * (so the linear walk skips them instead of desyncing). ANY mismatch returns 0
 * and the caller keeps the blanket has_indirect decline — the safe floor.
 * Index bound: cmp+ja when present; else a `movzx rI, r/m8` defining the index
 * within the window bounds it to 256. No bound => no resolution. ------------- */
#define JT_RING 8            /* lookback: cmp,ja,lea,movsxd,add precede the jmp */
#define JT_MAX_ENTRIES 16384
#define JT_MAX_TABLES 16
typedef struct { size_t lo, hi; } jt_range;

/* modrm/rex field helpers over a single-instruction byte view */
static int jt_rex(const uint8_t *p, int *rex){   /* optional single REX; no other prefixes */
    if ((p[0] & 0xF0) == 0x40) { *rex = p[0]; return 1; }
    *rex = 0; return 0;
}
static int resolve_jump_table(uint8_t *text, size_t fstart, size_t fend,
                              const size_t *ring, int nring, size_t jq, int jlen,
                              uint8_t *tgt, jt_range *tab, int *ntab) {
    if (nring < 3) return 0;
    /* jmp *rD : [rex] FF /4 mod=11 */
    { const uint8_t *p = text + jq; int rex, k = jt_rex(p, &rex);
      if (p[k] != 0xFF || ((p[k+1] >> 3) & 7) != 4 || (p[k+1] >> 6) != 3) return 0;
      if (k + 2 != jlen) return 0;
      int rD = (p[k+1] & 7) | ((rex & 1) << 3);

      /* prev1: add rD, rB  (01 /r: dst=rm ; 03 /r: dst=reg) */
      const uint8_t *a = text + ring[nring-1]; int arex, ak = jt_rex(a, &arex);
      int rB;
      if (a[ak] == 0x01 && (a[ak+1] >> 6) == 3) {
          int rm  = (a[ak+1] & 7)        | ((arex & 1) << 3);
          int reg = ((a[ak+1] >> 3) & 7) | ((arex & 4) << 1);
          if (rm != rD) return 0; rB = reg;
      } else if (a[ak] == 0x03 && (a[ak+1] >> 6) == 3) {
          int reg = ((a[ak+1] >> 3) & 7) | ((arex & 4) << 1);
          int rm  = (a[ak+1] & 7)        | ((arex & 1) << 3);
          if (reg != rD) return 0; rB = rm;
      } else return 0;

      /* prev2: movsxd rD, [rB + rI*4]  (63 /r mod=00 rm=100, SIB scale=2) */
      const uint8_t *m = text + ring[nring-2]; int mrex, mk = jt_rex(m, &mrex);
      if (m[mk] != 0x63 || (m[mk+1] >> 6) != 0 || (m[mk+1] & 7) != 4) return 0;
      { int reg = ((m[mk+1] >> 3) & 7) | ((mrex & 4) << 1);
        if (reg != rD) return 0; }
      if ((m[mk+2] >> 6) != 2) return 0;                       /* scale = *4 */
      int rI = ((m[mk+2] >> 3) & 7) | ((mrex & 2) << 2);
      { int base = (m[mk+2] & 7) | ((mrex & 1) << 3);
        if (base != rB || (m[mk+2] & 7) == 5) return 0; }      /* base==101 needs disp */

      /* prev3: lea rB, [rip+d32]  (8D /r mod=00 rm=101) -> table offset */
      const uint8_t *l = text + ring[nring-3]; int lrex, lk = jt_rex(l, &lrex);
      if (l[lk] != 0x8D || (l[lk+1] >> 6) != 0 || (l[lk+1] & 7) != 5) return 0;
      { int reg = ((l[lk+1] >> 3) & 7) | ((lrex & 4) << 1);
        if (reg != rB) return 0; }
      if (rB == rI || rB == rD) return 0;   /* aliased base would corrupt the pattern */
      int32_t d32; memcpy(&d32, l + lk + 2, 4);
      size_t lea_end = ring[nring-3] + (size_t)lk + 6;
      long tbl_l = (long)lea_end + d32;
      if (tbl_l < (long)fstart || (size_t)tbl_l >= fend) return 0;   /* table must be in-function */
      size_t tbl = (size_t)tbl_l;

      /* Index bound: `cmp rI,imm` guarded by a forward ja past the jmp, or a
       * `movzx rI, r/m8` (bounds the index to 0..255). SOUNDNESS: every ring
       * instruction between the bounding insn and the movsxd must be one we
       * fully understand as not writing rI — only the guarding ja and the lea
       * (which writes rB != rI) qualify. Anything else => unproven => fail. */
      long n_ent = -1; int ja_seen = 0;
      for (int h = nring - 4; h >= 0; h--) {
          const uint8_t *c = text + ring[h]; int crex, ck = jt_rex(c, &crex);
          /* guarding ja: part of the gap; note whether it jumps past the dispatch */
          if (c[0] == 0x77) {
              if ((long)ring[h] + 2 + (int8_t)c[1] > (long)jq) ja_seen = 1;
              continue;
          }
          if (c[0] == 0x0F && c[1] == 0x87) {
              int32_t v; memcpy(&v, c + 2, 4);
              if ((long)ring[h] + 6 + v > (long)jq) ja_seen = 1;
              continue;
          }
          long imm = -1;
          if (c[ck] == 0x83 && (c[ck+1] >> 6) == 3 && ((c[ck+1] >> 3) & 7) == 7
              && ((c[ck+1] & 7) | ((crex & 1) << 3)) == rI)
              imm = (int8_t)c[ck+2];
          else if (c[ck] == 0x81 && (c[ck+1] >> 6) == 3 && ((c[ck+1] >> 3) & 7) == 7
                   && ((c[ck+1] & 7) | ((crex & 1) << 3)) == rI)
              { int32_t v; memcpy(&v, c + ck + 2, 4); imm = v; }
          else if (c[ck] == 0x3D && rI == 0)
              { int32_t v; memcpy(&v, c + ck + 1, 4); imm = v; }
          else if (c[ck] == 0x0F && c[ck+1] == 0xB6
                   && (((c[ck+2] >> 3) & 7) | ((crex & 4) << 1)) == rI)
              { n_ent = 256; break; }        /* movzx 8-bit index: bound needs no ja */
          if (imm < 0) return 0;             /* unknown insn in the gap -> unproven */
          if (!ja_seen || imm < 0) return 0; /* cmp without a guarding ja -> unproven */
          n_ent = imm + 1;
          break;
      }
      if (n_ent <= 0 || n_ent > JT_MAX_ENTRIES) return 0;
      if (tbl + 4u * (size_t)n_ent > fend) return 0;           /* table must fit in-function */
      if (*ntab >= JT_MAX_TABLES) return 0;

      /* mark every in-function target; record the table as data to skip */
      for (long i = 0; i < n_ent; i++) {
          int32_t e; memcpy(&e, text + tbl + 4 * (size_t)i, 4);
          long t = (long)tbl + e;
          if (t >= (long)fstart && t < (long)fend) tgt[t - fstart] = 1;
      }
      tab[*ntab].lo = tbl; tab[*ntab].hi = tbl + 4u * (size_t)n_ent; (*ntab)++;
      return 1;
    }
}

/* Pass-1 branch-target collection over one function [fstart,fend) (offsets into
 * text). Marks tgt[off-fstart]=1 at every in-function direct branch target, and
 * sets *has_indirect if any UNRESOLVED indirect jmp is seen (bounded jump-table
 * dispatches are resolved: their targets are marked and their inline table bytes
 * skipped as data). Returns 1 if the function decoded cleanly to its boundary, 0
 * on a length-decode desync (caller treats a 0 conservatively). Shared by the
 * eager scanner (scan_function) and the runtime patch-safety check
 * (avxemu_patch_safe). tgt[] must be caller-allocated, length >= fend-fstart and
 * pre-zeroed. */
static int collect_branch_targets(uint8_t *text, size_t fstart, size_t fend,
                                  uint8_t *tgt, int *has_indirect) {
    *has_indirect = 0;
    size_t ring[JT_RING]; int nring = 0;
    jt_range tab[JT_MAX_TABLES]; int ntab = 0;
    size_t q = fstart;
    while (q < fend) {
        int skipped = 0;
        for (int i = 0; i < ntab; i++)
            if (q >= tab[i].lo && q < tab[i].hi) { q = tab[i].hi; skipped = 1; break; }
        if (skipped) { nring = 0; continue; }              /* table data: not code */
        if (fend - q <= 15) { int pad = 1; for (size_t r = q; r < fend; r++) if (text[r]&&text[r]!=0xCC&&text[r]!=0x90){pad=0;break;} if (pad) break; }
        int zk, o2; int len = x86_len(text + q, text + fend, &zk, &o2);
        if (len <= 0) return 0;
        int term; long t; int ind;
        lde_cflow(text + q, text + fend, len, (long)q, &term, &t, &ind);
        if (ind && !resolve_jump_table(text, fstart, fend, ring, nring, q, len, tgt, tab, &ntab))
            *has_indirect = 1;
        if (t >= (long)fstart && t < (long)fend) tgt[t - fstart] = 1;
        if (nring == JT_RING) { memmove(ring, ring + 1, (JT_RING - 1) * sizeof ring[0]); nring--; }
        ring[nring++] = q;
        q += len;
    }
    return 1;
}

/* ---- function-bounds cache for the runtime patch-safety check (Part 3). Filled
 * once by install_trampolines from the SAME LC_FUNCTION_STARTS data the eager scan
 * walks, so avxemu_patch_safe is cheap per call. g_func_starts holds ABSOLUTE
 * addresses (vmaddr+slide) of every function start inside __text, sorted ascending;
 * the function containing an address A is [start_i, start_{i+1}) where start_i is
 * the greatest start <= A (the last function ends at g_text_hi). ----------------- */
static uint64_t *g_func_starts; static int g_func_count;
static uint8_t  *g_text_base; static uint64_t g_text_lo, g_text_hi;

/* Branch-target scratch for the patch-safety scan. avxemu_patch_safe now runs
 * INSIDE the SIGILL handler (on_sigill -> avxemu_relocate_block -> avxemu_patch_safe),
 * so it must NOT call malloc/free: those are not async-signal-safe and the fault
 * could have interrupted the allocator mid-update, risking deadlock. A fixed static
 * map sized to a generous max function span replaces the heap allocation. If a
 * containing function's span exceeds the cap we DECLINE (return 0, the safe floor).
 * SINGLE-THREADED: this map is touched only during a patch attempt; the startup
 * spin this targets is single-threaded main-thread, so concurrent relocation
 * attempts from multiple threads are out of scope for Milestone A — no locking. */
#define PATCHSAFE_MAP_MAX (256u * 1024)
static uint8_t g_patchsafe_map[PATCHSAFE_MAP_MAX];

/* Is it safe to write a `jmplen`-byte jmp at `site`? Decline unless we can PROVE
 * no branch in the containing function targets the open interval (site, site+jmplen).
 * Conservative: decline if the layout is unknown, the site is outside scanned text,
 * the function won't decode cleanly, or it contains an unresolved indirect jmp
 * (whose target could land anywhere, including inside the jmp footprint). */
int avxemu_patch_safe(uint8_t *site, int jmplen) {
    if (!g_func_starts || g_func_count <= 0 || !g_text_base) return 0;  /* layout unknown */
    if (jmplen <= 1) return 1;
    uint64_t a = (uint64_t)(uintptr_t)site;
    if (a < g_text_lo || a >= g_text_hi) return 0;                      /* outside scanned text */

    /* greatest function start <= a */
    int lo = 0, hi = g_func_count - 1, idx = -1;
    while (lo <= hi) { int mid = (lo + hi) >> 1;
        if (g_func_starts[mid] <= a) { idx = mid; lo = mid + 1; } else hi = mid - 1; }
    if (idx < 0) return 0;
    uint64_t fstart_abs = g_func_starts[idx];
    uint64_t fend_abs   = (idx + 1 < g_func_count) ? g_func_starts[idx + 1] : g_text_hi;
    if (fend_abs > g_text_hi) fend_abs = g_text_hi;
    if (a + (uint64_t)jmplen > fend_abs) return 0;     /* jmp would spill past function end */

    size_t fstart = (size_t)(fstart_abs - g_text_lo);
    size_t fend   = (size_t)(fend_abs   - g_text_lo);
    size_t n = fend - fstart;
    if (n > PATCHSAFE_MAP_MAX) return 0;               /* span exceeds scratch cap -> decline (safe floor) */
    uint8_t *tgt = g_patchsafe_map;                    /* static scratch: no malloc in the handler */
    memset(tgt, 0, n);                                 /* collect_branch_targets needs a pre-zeroed map */
    int has_indirect = 0;
    int clean = collect_branch_targets(g_text_base, fstart, fend, tgt, &has_indirect);
    int safe = 1;
    if (!clean || has_indirect) {
        safe = 0;                                      /* can't fully account for targets */
    } else {
        size_t soff = (size_t)(a - g_text_lo) - fstart;  /* site offset within function */
        for (int k = 1; k < jmplen; k++) {               /* open interval (site, site+jmplen) */
            size_t off = soff + (size_t)k;
            if (off < n && tgt[off]) { safe = 0; break; }
        }
    }
    return safe;
}

/* Test hook (reloctest): register a single synthetic function [base, base+size)
 * as the cached layout so avxemu_patch_safe can run against a unit-test buffer not
 * in the main image's __text. Production NEVER calls this — install_trampolines
 * populates the cache from LC_FUNCTION_STARTS. */
void avxemu_patch_safe_test_region(uint8_t *base, size_t size) {
    static uint64_t one_start;
    one_start = (uint64_t)(uintptr_t)base;
    g_func_starts = &one_start; g_func_count = 1;
    g_text_base = base;
    g_text_lo = (uint64_t)(uintptr_t)base;
    g_text_hi = g_text_lo + size;
}

/*
 * Trampoline one function's faulting runs. A function that decodes cleanly
 * linearly is walked straight through; one that doesn't (embedded jump tables)
 * is mapped by recursive descent so its reachable code is covered too instead of
 * being skipped wholesale. Faulting (AVX2/BMI/FMA) instructions never branch, so
 * a run of physically-consecutive ones always falls through to real code — the
 * resume point is safe in both modes.
 */
static long scan_function(uint8_t *text, size_t fstart, size_t fend, size_t readable) {
    size_t n = fend - fstart;
    uint8_t *tgt = calloc(n, 1);
    if (!tgt) return 0;
    long patched = 0;

    /* pass 1: clean linear decode? collect direct-branch targets + indirect flag */
    int has_indirect = 0;
    int clean = collect_branch_targets(text, fstart, fend, tgt, &has_indirect);

    if (clean) {
        /* pass 2: walk linearly; trampoline each maximal faulting run */
        size_t q = fstart;
        while (q < fend) {
            if (fend - q <= 15) { int pad = 1; for (size_t r = q; r < fend; r++) if (text[r]&&text[r]!=0xCC&&text[r]!=0x90){pad=0;break;} if (pad) break; }
            int zk, o2; int len = x86_len(text + q, text + fend, &zk, &o2);
            if (len <= 0) break;
            decoded d; int dl = decode(text + q, &d);
            if (!(dl > 0 && d.op && tramp_faults(&d))) { q += len; continue; }
            tramp_insn insns[MAXRUN]; size_t offs[MAXRUN]; int ni;
            size_t re = gather_run(text, fstart, fend, q, 0, insns, offs, &ni);
            patched += emit_run(text, fstart, fend, insns, offs, ni, re, tgt, has_indirect);
            q = re;
        }
    } else {
        /* dirty: recursive-descent reachability map, then trampoline faulting runs
         * inside reachable code only (jump-table data is never visited or patched). */
        uint8_t *code = calloc(n, 1);
        memset(tgt, 0, n);                               /* linear targets past the desync are bogus */
        if (code && lde_rd_map(text, fstart, fend, readable, code, tgt, &has_indirect)) {
            size_t off = 0;
            while (off < n) {
                if (!code[off]) { off++; continue; }
                size_t qq = fstart + off;
                int zk, o2; int len = x86_len(text + qq, text + fend, &zk, &o2);
                decoded d; int dl = (len > 0) ? decode(text + qq, &d) : 0;
                if (!(dl > 0 && d.op && tramp_faults(&d))) { off++; continue; }
                tramp_insn insns[MAXRUN]; size_t offs[MAXRUN]; int ni;
                size_t re = gather_run(text, fstart, fend, qq, code, insns, offs, &ni);
                patched += emit_run(text, fstart, fend, insns, offs, ni, re, tgt, has_indirect);
                off = re - fstart;
            }
        }
        free(code);
    }
    free(tgt);
    return patched;
}

static uint64_t uleb_t(const uint8_t **p, const uint8_t *e) {
    uint64_t r = 0; int s = 0; uint8_t x;
    do { if (*p >= e) return r; x = *(*p)++; r |= (uint64_t)(x & 0x7f) << s; s += 7; } while (x & 0x80);
    return r;
}

/* Install trampolines over the main executable. Returns jmps written. */
long avxemu_install_trampolines(void) {
    detect_features();
    if (!g_lack_avx2 && !g_lack_fma && !g_lack_bmi && !g_lack_f16c) return 0;  /* nothing faults here */

    /* Diagnostics (env-gated): resolve flags now so the hot path sees them, and
     * arm the SIGUSR2 / atexit histogram dump if AVXEMU_OPHIST is set. */
    (void)nstats_enabled();
    if (ophist_enabled()){
        struct sigaction sa; memset(&sa, 0, sizeof sa);
        sa.sa_handler = ophist_sigusr2; sigemptyset(&sa.sa_mask);
        sigaction(SIGUSR2, &sa, 0);
        atexit(ophist_dump);
    }

    /* per-thread side-stack key, created here (single-threaded) before any thunk runs */
    if (!g_side_key_ok && pthread_key_create(&g_side_key, 0) == 0) g_side_key_ok = 1;

    const struct mach_header_64 *mh = (const struct mach_header_64 *)_dyld_get_image_header(0);
    if (!mh || mh->magic != MH_MAGIC_64) return 0;
    intptr_t slide = _dyld_get_image_vmaddr_slide(0);

    uint64_t tseg_vm = 0, tseg_sz = 0, text_addr = 0, text_size = 0, le_vm = 0, le_off = 0;
    uint32_t fs_off = 0, fs_size = 0;
    const struct load_command *lc = (const struct load_command *)(mh + 1);
    for (uint32_t i = 0; i < mh->ncmds; i++) {
        if (lc->cmd == LC_SEGMENT_64) {
            const struct segment_command_64 *sg = (const struct segment_command_64 *)lc;
            if (!strcmp(sg->segname, "__TEXT")) {
                tseg_vm = sg->vmaddr; tseg_sz = sg->vmsize;
                const struct section_64 *sc = (const struct section_64 *)(sg + 1);
                for (uint32_t j = 0; j < sg->nsects; j++)
                    if (!strcmp(sc[j].sectname, "__text")) { text_addr = sc[j].addr; text_size = sc[j].size; }
            } else if (!strcmp(sg->segname, "__LINKEDIT")) { le_vm = sg->vmaddr; le_off = sg->fileoff; }
        } else if (lc->cmd == LC_FUNCTION_STARTS) {
            const struct linkedit_data_command *ld = (const struct linkedit_data_command *)lc;
            fs_off = ld->dataoff; fs_size = ld->datasize;
        }
        lc = (const struct load_command *)((const char *)lc + lc->cmdsize);
    }
    if (!text_addr || !fs_off) return 0;

    uint8_t *text = (uint8_t *)(text_addr + slide);
    /* switch-table resolution may read entries in __const after __text */
    size_t readable = (size_t)((tseg_vm + tseg_sz) - text_addr);

    /* thunk pool, placed near __text so jmp rel32 reaches it */
    size_t pool_sz = 96u << 20;
    void *hint = (void *)(((uintptr_t)(text + text_size) + 0x100000) & ~(uintptr_t)0xfff);
    if (!avxemu_pool_init(hint, pool_sz)) return 0;

    /* make __text writable for the patch pass (COPY: __TEXT maxprot lacks write) */
    mach_port_t task = mach_task_self();
    uintptr_t lo = (uintptr_t)text & ~(uintptr_t)0xfff;
    uintptr_t hi = ((uintptr_t)text + text_size + 0xfff) & ~(uintptr_t)0xfff;
    if (vm_protect(task, (vm_address_t)lo, (vm_size_t)(hi - lo), FALSE,
                   VM_PROT_READ | VM_PROT_WRITE | VM_PROT_COPY) != KERN_SUCCESS)
        return 0;

    const uint8_t *fp = (const uint8_t *)(le_vm + slide + (fs_off - le_off));
    const uint8_t *fe = fp + fs_size;

    /* Cache function bounds for the runtime patch-safety check (Part 3) from the
     * SAME LC_FUNCTION_STARTS data the scan loop below walks. Two cheap passes over
     * the (copied) cursor — count in-__text starts, then record absolute addresses.
     * fp is left untouched for the scan loop. */
    g_text_base = text;
    g_text_lo = (uint64_t)(uintptr_t)text;
    g_text_hi = g_text_lo + text_size;
    {
        const uint8_t *cp = fp; uint64_t ca = tseg_vm; int cnt = 0;
        while (cp < fe) { uint64_t d = uleb_t(&cp, fe); if (!d) break; ca += d;
            if (ca >= text_addr && ca < text_addr + text_size) cnt++; }
        if (cnt > 0) {
            g_func_starts = (uint64_t *)malloc((size_t)cnt * sizeof(uint64_t));
            if (g_func_starts) {
                const uint8_t *cp2 = fp; uint64_t ca2 = tseg_vm; int k = 0;
                while (cp2 < fe && k < cnt) { uint64_t d = uleb_t(&cp2, fe); if (!d) break; ca2 += d;
                    if (ca2 >= text_addr && ca2 < text_addr + text_size)
                        g_func_starts[k++] = ca2 + (uint64_t)slide; }   /* absolute, ascending */
                g_func_count = k;
            }
        }
    }

    uint64_t addr = tseg_vm; uint64_t prev = 0; long total = 0;
    while (fp < fe) {
        uint64_t delta = uleb_t(&fp, fe); if (!delta) break;
        addr += delta;
        if (prev && prev >= text_addr && prev < text_addr + text_size) {
            uint64_t fend = addr; if (fend > text_addr + text_size) fend = text_addr + text_size;
            total += scan_function(text, prev - text_addr, fend - text_addr, readable);
        }
        prev = addr;
    }
    if (prev && prev >= text_addr && prev < text_addr + text_size) {
        total += scan_function(text, prev - text_addr, text_size, readable);
    }

    vm_protect(task, (vm_address_t)lo, (vm_size_t)(hi - lo), FALSE, VM_PROT_READ | VM_PROT_EXECUTE);

    if (g_nstats){   /* AVXEMU_NATIVE_STATS run-composition report (stderr) */
        emit("=== AVXEMU_NATIVE_STATS (install-time run composition) ===\n");
        emit("native-accept runs\t"); as_u64(2, g_ns_accept); emit("\n");
        emit("native-decline runs\t"); as_u64(2, g_ns_decline); emit("\n");
        emit("single-insn runs\t"); as_u64(2, g_ns_single); emit("\n");
        emit("multi-insn runs\t"); as_u64(2, g_ns_multi); emit("\n");
        int idx[VEX_OP_COUNT]; int m = 0;
        for (int i = 0; i < VEX_OP_COUNT; i++) if (g_ns_block_op[i]) idx[m++] = i;
        for (int i = 0; i < m; i++){ int best = i;
            for (int j = i + 1; j < m; j++) if (g_ns_block_op[idx[j]] > g_ns_block_op[idx[best]]) best = j;
            int t = idx[i]; idx[i] = idx[best]; idx[best] = t; }
        emit("-- first-blocking op in declined runs (desc) --\n");
        for (int i = 0; i < m; i++){ const char *nm = vex_op_name((vex_op)idx[i]);
            emit(nm); emit("\t"); as_u64(2, g_ns_block_op[idx[i]]); emit("\n"); }
    }
    return total;
}
