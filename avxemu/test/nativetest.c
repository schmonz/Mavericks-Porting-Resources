/*
 * nativetest.c — differential oracle for the register-resident native-SSE
 * codegen thunk (Milestone B).
 *
 * For each supported op (register form AND memory-operand form) and several
 * multi-op runs with data dependencies, we:
 *   1. build a run_record of hand-constructed `decoded` instructions (built
 *      directly, not from bytes — the emitter and the C emulator both consume
 *      the decoded struct, so this avoids fragile VEX encoding while exercising
 *      exactly the fields production decode() fills);
 *   2. run the GROUND TRUTH path avxemu_tramp_dispatch() (C emulate via vec_exec)
 *      against a seeded regfile + backing memory;
 *   3. run the EMITTED native block avxemu_emit_native_block() against an
 *      IDENTICAL regfile + the same backing memory;
 *   4. assert rf->ymm is byte-identical across all 16 registers.
 *
 * Includes wide (256-bit) and 128-bit (VEX.128 zero-upper) forms, memory
 * operands (base, base+index*scale, rip-relative), and a wide+mem multi-op run
 * with a data dependency through the register file.
 *
 * Hermetic: builds with the SSE-only core flags and never executes a VEX
 * instruction (the decoded structs are synthetic; the C path is vec_exec, the
 * native block is SSE2/SSE4.1). Runs on a no-AVX2 host.
 */
#include "decode.h"
#include "regfile.h"
#include "vexops.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/mman.h>

/* Mirror tramp.c's private layout (run_record appended after a thunk). */
typedef struct { uint64_t addr; decoded dec; } tramp_insn;
typedef struct { uint32_t n; uint32_t pad; tramp_insn insns[]; } run_record;

extern void  avxemu_tramp_dispatch(const void *r, void *rf);   /* C ground truth (vector) */
extern void *avxemu_emit_native_block(const void *insns, int n);
extern void  avxemu_tramp_dispatch_bmi(const void *r, void *rf);   /* C ground truth (scalar BMI) */
extern void *avxemu_emit_native_block_bmi(const void *insns, int n);
extern int   avxemu_pool_init(void *hint, size_t cap);

/* RFLAGS bit positions (mirror regs.h; the test only needs these). */
#define TF_CF (1ull<<0)
#define TF_PF (1ull<<2)
#define TF_AF (1ull<<4)
#define TF_ZF (1ull<<6)
#define TF_SF (1ull<<7)
#define TF_OF (1ull<<11)
#define TF_OWNED (TF_CF|TF_ZF|TF_SF|TF_OF)

static int g_fail;
static uint8_t *g_mem;          /* page-backed memory operand source */
#define MEMSZ 4096

/* deterministic fill so both runs see identical inputs */
static void seed_rf(avxemu_regfile *rf, unsigned s){
    memset(rf, 0, sizeof *rf);
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 32; j++){ s = s * 1664525u + 1013904223u; rf->ymm[i][j] = (uint8_t)(s >> 23); }
    rf->rflags = 0x202;
}

typedef struct { int reg; uint64_t val; } gprset;

static int run_case(const char *name, tramp_insn *insns, int n,
                    const gprset *gs, int ng){
    size_t rsz = sizeof(uint32_t) * 2 + (size_t)n * sizeof(tramp_insn);
    run_record *rec = (run_record *)malloc(rsz);
    rec->n = (uint32_t)n; rec->pad = 0;
    for (int i = 0; i < n; i++) rec->insns[i] = insns[i];

    avxemu_regfile rfc, rfn;
    seed_rf(&rfc, 0x9E37u + (unsigned)(unsigned char)name[0] * 131u);
    memcpy(&rfn, &rfc, sizeof rfc);
    for (int i = 0; i < ng; i++){ rfc.gpr[gs[i].reg] = gs[i].val; rfn.gpr[gs[i].reg] = gs[i].val; }

    avxemu_tramp_dispatch(rec, &rfc);                       /* (a) ground truth */

    void *blk = avxemu_emit_native_block(rec->insns, n);
    if (!blk){ printf("  %-46s emit returned NULL — FAIL\n", name); g_fail++; free(rec); return 1; }
    ((void (*)(const void *, void *))blk)(rec, &rfn);       /* (b) native block */

    int bad = memcmp(rfc.ymm, rfn.ymm, sizeof rfc.ymm) != 0;
    if (bad){
        for (int i = 0; i < 16 && bad; i++)
            if (memcmp(rfc.ymm[i], rfn.ymm[i], 32)){
                printf("  %-46s ymm%d MISMATCH\n", name, i);
                printf("      C  :"); for (int j = 0; j < 32; j++) printf(" %02x", rfc.ymm[i][j]); printf("\n");
                printf("      NAT:"); for (int j = 0; j < 32; j++) printf(" %02x", rfn.ymm[i][j]); printf("\n");
                break;
            }
        g_fail++;
    }
    printf("  %-46s %s\n", name, bad ? "FAIL" : "ok");
    free(rec);
    return bad;
}

/* ---- decoded constructors (faithful to decode.c field assignment) ---- */
static decoded mk_vvv(vex_op op, int dst, int a, int b, int wide){
    decoded d; memset(&d, 0, sizeof d);
    d.op = op; d.dst = dst; d.dst_kind = DST_YMM; d.a_src = (int8_t)a; d.b_src = (int8_t)b;
    d.c_src = OPND_NONE; d.wide = (uint8_t)wide; d.len = 4;
    d.base = OPND_NONE; d.index = OPND_NONE; d.scale = 1; d.bmi_dst2 = OPND_NONE;
    return d;
}
static decoded mk_vvv_mem(vex_op op, int dst, int a, int wide,
                          int base, int index, int scale, int32_t disp, int rip_rel){
    decoded d = mk_vvv(op, dst, a, OPND_MEM, wide);
    d.has_mem = 1; d.base = (int8_t)base; d.index = (int8_t)index;
    d.scale = (uint8_t)scale; d.disp = disp; d.rip_rel = (uint8_t)rip_rel;
    d.mem_bytes = (uint8_t)(wide ? 32 : 16); d.len = 6;
    return d;
}
static decoded mk_unb(vex_op op, int dst, int b, int wide, int membytes){
    decoded d; memset(&d, 0, sizeof d);
    d.op = op; d.dst = dst; d.dst_kind = DST_YMM; d.a_src = OPND_NONE; d.b_src = (int8_t)b;
    d.c_src = OPND_NONE; d.wide = (uint8_t)wide; d.len = 5;
    d.base = OPND_NONE; d.index = OPND_NONE; d.scale = 1; d.bmi_dst2 = OPND_NONE;
    d.mem_bytes = (uint8_t)membytes;
    return d;
}
static decoded mk_unb_mem(vex_op op, int dst, int wide, int membytes,
                          int base, int index, int scale, int32_t disp, int rip_rel){
    decoded d = mk_unb(op, dst, OPND_MEM, wide, membytes);
    d.has_mem = 1; d.base = (int8_t)base; d.index = (int8_t)index;
    d.scale = (uint8_t)scale; d.disp = disp; d.rip_rel = (uint8_t)rip_rel; d.len = 6;
    return d;
}
static tramp_insn TI(decoded d){ tramp_insn t; t.addr = 0x400000; t.dec = d; return t; }

/* LZCNT decoded: dst<-reg, a_src<-rm (the source). Register operands only. */
static decoded mk_lz(int dst, int src, int opsize){
    decoded d; memset(&d, 0, sizeof d);
    d.op = BMI_LZCNT; d.is_bmi = 1; d.opsize = (uint8_t)opsize;
    d.dst = (int8_t)dst; d.dst_kind = DST_GPR;
    d.a_src = (int8_t)src; d.b_src = OPND_NONE; d.c_src = OPND_NONE;
    d.base = OPND_NONE; d.index = OPND_NONE; d.scale = 1; d.bmi_dst2 = OPND_NONE;
    d.mem_bytes = (uint8_t)(opsize/8); d.len = 4;
    return d;
}

/* Scalar-GPR differential driver: run the C BMI ground truth and the emitted
 * native scalar block on IDENTICAL regfiles (same gpr seed + rflags), then assert
 * every GPR slot and rflags match bit-for-bit, that the owned flag bits (CF/ZF/
 * SF/OF) agree, and that PF/AF are preserved from the input flags. */
static int run_case_bmi(const char *name, tramp_insn *insns, int n,
                        const gprset *gs, int ng, uint64_t flags_in){
    size_t rsz = sizeof(uint32_t) * 2 + (size_t)n * sizeof(tramp_insn);
    run_record *rec = (run_record *)malloc(rsz);
    rec->n = (uint32_t)n; rec->pad = 0;
    for (int i = 0; i < n; i++) rec->insns[i] = insns[i];

    avxemu_regfile rfc, rfn;
    seed_rf(&rfc, 0x1234u + (unsigned)(unsigned char)name[0] * 131u);
    memcpy(&rfn, &rfc, sizeof rfc);
    rfc.rflags = rfn.rflags = flags_in;
    for (int i = 0; i < ng; i++){ rfc.gpr[gs[i].reg] = gs[i].val; rfn.gpr[gs[i].reg] = gs[i].val; }

    avxemu_tramp_dispatch_bmi(rec, &rfc);                   /* (a) ground truth (bmi_exec) */

    void *blk = avxemu_emit_native_block_bmi(rec->insns, n);
    if (!blk){ printf("  %-46s emit returned NULL — FAIL\n", name); g_fail++; free(rec); return 1; }
    ((void (*)(const void *, void *))blk)(rec, &rfn);       /* (b) native scalar block */

    int bad = 0;
    if (memcmp(rfc.gpr, rfn.gpr, sizeof rfc.gpr)){
        for (int i = 0; i < 16; i++) if (rfc.gpr[i] != rfn.gpr[i])
            printf("  %-46s gpr%d C=%016llx NAT=%016llx\n", name, i,
                   (unsigned long long)rfc.gpr[i], (unsigned long long)rfn.gpr[i]);
        bad = 1;
    }
    if (rfc.rflags != rfn.rflags){
        printf("  %-46s rflags C=%016llx NAT=%016llx (owned C=%03llx NAT=%03llx)\n", name,
               (unsigned long long)rfc.rflags, (unsigned long long)rfn.rflags,
               (unsigned long long)(rfc.rflags & TF_OWNED), (unsigned long long)(rfn.rflags & TF_OWNED));
        bad = 1;
    }
    if ((rfn.rflags & (TF_PF|TF_AF)) != (flags_in & (TF_PF|TF_AF))){
        printf("  %-46s PF/AF NOT preserved: in=%llx out=%llx\n", name,
               (unsigned long long)(flags_in & (TF_PF|TF_AF)),
               (unsigned long long)(rfn.rflags & (TF_PF|TF_AF)));
        bad = 1;
    }
    if (bad) g_fail++;
    printf("  %-46s %s\n", name, bad ? "FAIL" : "ok");
    free(rec);
    return bad;
}

int main(void){
    setvbuf(stdout, NULL, _IONBF, 0);
    if (!avxemu_pool_init(0, 4u << 20)){ printf("pool init failed\n"); return 2; }
    g_mem = (uint8_t *)mmap(0, MEMSZ, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (g_mem == MAP_FAILED){ printf("mem map failed\n"); return 2; }
    for (int i = 0; i < MEMSZ; i++) g_mem[i] = (uint8_t)(i * 7 + 3);

    uint64_t M = (uint64_t)(uintptr_t)g_mem;
    /* base reg holds &g_mem; an index reg for the scaled case (index*scale stays in-page) */
    gprset gmem[]  = { { 8, M } };                 /* r8 = base */
    gprset gidx[]  = { { 8, M }, { 9, 4 } };       /* r8 = base, r9 = index (index*8 = 32) */

    printf("== native-SSE codegen: emitted block vs C emulate (vec_exec) ==\n");

    /* ---- binary ALU ops: register form, wide (256-bit) ---- */
    { tramp_insn t[] = { TI(mk_vvv(VPAND,   5, 1, 2, 1)) }; run_case("vpand  ymm5,ymm1,ymm2 (wide,reg)",   t, 1, 0, 0); }
    { tramp_insn t[] = { TI(mk_vvv(VPOR,    5, 1, 2, 1)) }; run_case("vpor   ymm5,ymm1,ymm2 (wide,reg)",   t, 1, 0, 0); }
    { tramp_insn t[] = { TI(mk_vvv(VPXOR,   5, 1, 2, 1)) }; run_case("vpxor  ymm5,ymm1,ymm2 (wide,reg)",   t, 1, 0, 0); }
    { tramp_insn t[] = { TI(mk_vvv(VPSUBB,  5, 1, 2, 1)) }; run_case("vpsubb ymm5,ymm1,ymm2 (wide,reg)",   t, 1, 0, 0); }
    { tramp_insn t[] = { TI(mk_vvv(VPCMPEQB,5, 1, 2, 1)) }; run_case("vpcmpeqb ymm5,ymm1,ymm2 (wide,reg)", t, 1, 0, 0); }
    { tramp_insn t[] = { TI(mk_vvv(VPCMPGTB,5, 1, 2, 1)) }; run_case("vpcmpgtb ymm5,ymm1,ymm2 (wide,reg)", t, 1, 0, 0); }

    /* ---- binary ALU ops: 128-bit (VEX.128 zero-upper) register form ---- */
    { tramp_insn t[] = { TI(mk_vvv(VPAND,   5, 1, 2, 0)) }; run_case("vpand  xmm5,xmm1,xmm2 (128,reg,zupper)", t, 1, 0, 0); }
    { tramp_insn t[] = { TI(mk_vvv(VPXOR,   5, 1, 2, 0)) }; run_case("vpxor  xmm5,xmm1,xmm2 (128,reg,zupper)", t, 1, 0, 0); }
    { tramp_insn t[] = { TI(mk_vvv(VPCMPGTB,5, 1, 2, 0)) }; run_case("vpcmpgtb xmm5,xmm1,xmm2 (128,reg)",      t, 1, 0, 0); }

    /* ---- binary ALU ops: memory operand (src2), wide + 128 ---- */
    { tramp_insn t[] = { TI(mk_vvv_mem(VPAND,  5, 1, 1, 8, OPND_NONE, 1, 0, 0)) }; run_case("vpand  ymm5,ymm1,[r8] (wide,mem)",  t, 1, gmem, 1); }
    { tramp_insn t[] = { TI(mk_vvv_mem(VPSUBB, 5, 1, 1, 8, OPND_NONE, 1, 0, 0)) }; run_case("vpsubb ymm5,ymm1,[r8] (wide,mem)",  t, 1, gmem, 1); }
    { tramp_insn t[] = { TI(mk_vvv_mem(VPXOR,  5, 1, 0, 8, OPND_NONE, 1, 0, 0)) }; run_case("vpxor  xmm5,xmm1,[r8] (128,mem)",   t, 1, gmem, 1); }
    { tramp_insn t[] = { TI(mk_vvv_mem(VPCMPEQB,5,1, 1, 8, OPND_NONE, 1, 16, 0)) }; run_case("vpcmpeqb ymm5,ymm1,[r8+16] (mem,disp)", t, 1, gmem, 1); }
    { tramp_insn t[] = { TI(mk_vvv_mem(VPOR,   5, 1, 1, 8, 9, 8, 0, 0)) };          run_case("vpor   ymm5,ymm1,[r8+r9*8] (mem,sib)",  t, 1, gidx, 2); }

    /* rip-relative memory operand: addr+len+disp must resolve to g_mem. */
    {
        decoded d = mk_vvv_mem(VPAND, 5, 1, 1, OPND_NONE, OPND_NONE, 1, 0, 1);
        tramp_insn t0; t0.addr = (uint64_t)(uintptr_t)g_mem; t0.dec = d;     /* rip_next = g_mem+len */
        t0.dec.disp = -(int32_t)t0.dec.len;                                  /* EA = -len + (g_mem+len) = g_mem */
        tramp_insn t[] = { t0 };
        run_case("vpand  ymm5,ymm1,[rip] (wide,riprel)", t, 1, 0, 0);
    }

    /* ---- VPBROADCASTD: reg + mem, wide + 128 ---- */
    { tramp_insn t[] = { TI(mk_unb(VPBROADCASTD, 5, 2, 1, 4)) }; run_case("vpbroadcastd ymm5,xmm2 (wide,reg)", t, 1, 0, 0); }
    { tramp_insn t[] = { TI(mk_unb(VPBROADCASTD, 5, 2, 0, 4)) }; run_case("vpbroadcastd xmm5,xmm2 (128,reg)",  t, 1, 0, 0); }
    { tramp_insn t[] = { TI(mk_unb_mem(VPBROADCASTD, 5, 1, 4, 8, OPND_NONE, 1, 0, 0)) }; run_case("vpbroadcastd ymm5,[r8] (wide,mem)", t, 1, gmem, 1); }
    { tramp_insn t[] = { TI(mk_unb_mem(VPBROADCASTD, 5, 0, 4, 8, OPND_NONE, 1, 7, 0)) }; run_case("vpbroadcastd xmm5,[r8+7] (128,mem)", t, 1, gmem, 1); }

    /* ---- VPBROADCASTW: reg + mem, wide + 128 (broadcast word 0 -> 16/8 words) ---- */
    { tramp_insn t[] = { TI(mk_unb(VPBROADCASTW, 5, 2, 1, 2)) }; run_case("vpbroadcastw ymm5,xmm2 (wide,reg)", t, 1, 0, 0); }
    { tramp_insn t[] = { TI(mk_unb(VPBROADCASTW, 5, 2, 0, 2)) }; run_case("vpbroadcastw xmm5,xmm2 (128,reg)",  t, 1, 0, 0); }
    { tramp_insn t[] = { TI(mk_unb_mem(VPBROADCASTW, 5, 1, 2, 8, OPND_NONE, 1, 0, 0)) }; run_case("vpbroadcastw ymm5,[r8] (wide,mem)", t, 1, gmem, 1); }
    { tramp_insn t[] = { TI(mk_unb_mem(VPBROADCASTW, 5, 0, 2, 8, OPND_NONE, 1, 5, 0)) }; run_case("vpbroadcastw xmm5,[r8+5] (128,mem)", t, 1, gmem, 1); }
    { tramp_insn t[] = { TI(mk_unb_mem(VPBROADCASTW, 5, 1, 2, 8, 9, 8, 0, 0)) };         run_case("vpbroadcastw ymm5,[r8+r9*8] (wide,mem,sib)", t, 1, gidx, 2); }

    /* ---- VPMOVZXBW: reg + mem, wide (16B->16W) + 128 (8B->8W, zero upper) ---- */
    { tramp_insn t[] = { TI(mk_unb(VPMOVZXBW, 5, 2, 1, 16)) }; run_case("vpmovzxbw ymm5,xmm2 (wide,reg)", t, 1, 0, 0); }
    { tramp_insn t[] = { TI(mk_unb(VPMOVZXBW, 5, 2, 0, 16)) }; run_case("vpmovzxbw xmm5,xmm2 (128,reg)",  t, 1, 0, 0); }
    { tramp_insn t[] = { TI(mk_unb_mem(VPMOVZXBW, 5, 1, 16, 8, OPND_NONE, 1, 0, 0)) }; run_case("vpmovzxbw ymm5,[r8] (wide,mem)", t, 1, gmem, 1); }
    { tramp_insn t[] = { TI(mk_unb_mem(VPMOVZXBW, 5, 0, 16, 8, OPND_NONE, 1, 3, 0)) }; run_case("vpmovzxbw xmm5,[r8+3] (128,mem)", t, 1, gmem, 1); }

    printf("\n== multi-op runs with register-file data dependencies ==\n");
    /* chain: ymm5 = ymm1^ymm2 ; ymm6 = ymm5 & ymm3 ; ymm7 = ymm6 - ymm4 (wide) */
    { tramp_insn t[] = {
        TI(mk_vvv(VPXOR,  5, 1, 2, 1)),
        TI(mk_vvv(VPAND,  6, 5, 3, 1)),     /* reads ymm5 (prev dst) -> NX_CACHE reuse */
        TI(mk_vvv(VPSUBB, 7, 6, 4, 1)),     /* reads ymm6 (prev dst) */
      }; run_case("chain xor;and;subb (wide,deps)", t, 3, 0, 0); }

    /* 128-bit chain exercising zero-upper between dependent ops */
    { tramp_insn t[] = {
        TI(mk_vvv(VPOR,   5, 1, 2, 0)),
        TI(mk_vvv(VPAND,  6, 5, 3, 0)),
      }; run_case("chain or;and (128,deps,zupper)", t, 2, 0, 0); }

    /* wide + mem multi-op run with a dependency through the register file:
     *   ymm2 = broadcastd [r8]            (mem source)
     *   ymm3 = ymm2 & ymm1               (reads ymm2 = prev dst -> cache)
     *   ymm4 = ymm3 ^ [r8+16]            (mem source, reads ymm3 = prev dst)
     *   ymm5 = pmovzxbw [r8+1]           (mem source, wide) */
    { tramp_insn t[] = {
        TI(mk_unb_mem(VPBROADCASTD, 2, 1, 4, 8, OPND_NONE, 1, 0, 0)),
        TI(mk_vvv(VPAND, 3, 2, 1, 1)),
        TI(mk_vvv_mem(VPXOR, 4, 3, 1, 8, OPND_NONE, 1, 16, 0)),
        TI(mk_unb_mem(VPMOVZXBW, 5, 1, 16, 8, OPND_NONE, 1, 1, 0)),
      }; run_case("wide+mem multi-op (bcast;and;xor[m];zxbw[m])", t, 4, gmem, 1); }

    printf("\n== scalar-GPR native block: emitted vs C emulate (bmi_exec) ==\n");
    /* Input flags: PF|AF set (must be PRESERVED) and ALL owned bits (CF/ZF/SF/OF)
     * set to garbage (must be correctly recomputed). 0x202 = reserved+IF. */
    const uint64_t FIN = 0x202u | TF_PF | TF_AF | TF_CF | TF_ZF | TF_SF | TF_OF;

    /* opsize 32: zero input (->result=32, CF=1), 1, all-ones, msb-set, random.
     * dst reg slot preseeded with all-ones to verify 32-bit zero-extension. */
    { gprset g[] = { {1,0xFFFFFFFFFFFFFFFFull}, {2,0x00000000u} };           tramp_insn t[]={TI(mk_lz(1,2,32))}; run_case_bmi("lzcnt ecx,edx  src=0   (32,zero->32,CF)", t,1,g,2,FIN); }
    { gprset g[] = { {1,0xFFFFFFFFFFFFFFFFull}, {2,0x00000001u} };           tramp_insn t[]={TI(mk_lz(1,2,32))}; run_case_bmi("lzcnt ecx,edx  src=1   (32->31)",       t,1,g,2,FIN); }
    { gprset g[] = { {1,0xFFFFFFFFFFFFFFFFull}, {2,0xFFFFFFFFu} };           tramp_insn t[]={TI(mk_lz(1,2,32))}; run_case_bmi("lzcnt ecx,edx  all-ones(32->0,ZF)",     t,1,g,2,FIN); }
    { gprset g[] = { {1,0xFFFFFFFFFFFFFFFFull}, {2,0x80000000u} };           tramp_insn t[]={TI(mk_lz(1,2,32))}; run_case_bmi("lzcnt ecx,edx  msb-set (32->0,ZF)",     t,1,g,2,FIN); }
    { gprset g[] = { {1,0xFFFFFFFFFFFFFFFFull}, {2,0x00 + 0x00123456u} };    tramp_insn t[]={TI(mk_lz(1,2,32))}; run_case_bmi("lzcnt ecx,edx  random  (32)",           t,1,g,2,FIN); }
    /* opsize 32, high bits in src reg must be ignored (masked to 32) */
    { gprset g[] = { {1,0xFFFFFFFFFFFFFFFFull}, {2,0xDEADBEEF00000010ull} }; tramp_insn t[]={TI(mk_lz(1,2,32))}; run_case_bmi("lzcnt ecx,edx  src hi-garbage (32 mask)", t,1,g,2,FIN); }

    /* opsize 64: zero, 1, all-ones, msb-set, random */
    { gprset g[] = { {2,0ull} };                  tramp_insn t[]={TI(mk_lz(1,2,64))}; run_case_bmi("lzcnt rcx,rdx  src=0   (64,zero->64,CF)", t,1,g,1,FIN); }
    { gprset g[] = { {2,1ull} };                  tramp_insn t[]={TI(mk_lz(1,2,64))}; run_case_bmi("lzcnt rcx,rdx  src=1   (64->63)",       t,1,g,1,FIN); }
    { gprset g[] = { {2,0xFFFFFFFFFFFFFFFFull} }; tramp_insn t[]={TI(mk_lz(1,2,64))}; run_case_bmi("lzcnt rcx,rdx  all-ones(64->0,ZF)",     t,1,g,1,FIN); }
    { gprset g[] = { {2,0x8000000000000000ull} };tramp_insn t[]={TI(mk_lz(1,2,64))}; run_case_bmi("lzcnt rcx,rdx  msb-set (64->0,ZF)",     t,1,g,1,FIN); }
    { gprset g[] = { {2,0x0000123456789abcull} };tramp_insn t[]={TI(mk_lz(1,2,64))}; run_case_bmi("lzcnt rcx,rdx  random  (64)",           t,1,g,1,FIN); }

    /* dst==src aliasing (lzcnt eax,eax / rax,rax) */
    { gprset g[] = { {0,0x00000040u} };           tramp_insn t[]={TI(mk_lz(0,0,32))}; run_case_bmi("lzcnt eax,eax  dst==src (32)",          t,1,g,1,FIN); }
    { gprset g[] = { {0,0ull} };                  tramp_insn t[]={TI(mk_lz(0,0,64))}; run_case_bmi("lzcnt rax,rax  dst==src src=0 (64)",    t,1,g,1,FIN); }
    /* high-numbered regs (REX paths) */
    { gprset g[] = { {11,0x000000000000ff00ull} };tramp_insn t[]={TI(mk_lz(10,11,64))}; run_case_bmi("lzcnt r10,r11  (64,REX)",            t,1,g,1,FIN); }

    /* multi-op scalar run (two LZCNTs; second reads first's result via the slots) */
    { gprset g[] = { {2,0x00000ff0u} };
      tramp_insn t[]={ TI(mk_lz(1,2,32)), TI(mk_lz(3,1,32)) };
      run_case_bmi("lzcnt ecx,edx ; lzcnt ebx,ecx (multi)", t,2,g,1,FIN); }

    printf("\nNATIVETEST TOTAL: %d failure(s)\n", g_fail);
    return g_fail ? 1 : 0;
}
