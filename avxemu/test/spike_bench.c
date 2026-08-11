/*
 * spike_bench.c -- Milestone B go/no-go micro-benchmark.
 *
 * THROWAWAY measurement spike. Question: would translating a *run* of trampolined
 * AVX2 vector ops into a register-resident SSE block collapse the no-AVX2 startup
 * spin, vs the current per-instruction `avxemu_emulate` dispatch path?
 *
 * The live spin (dtrace, 2026-06-30 RULED-OUT entry) is NOT trap-bound: it is the
 * eager-trampoline path calling `avxemu_emulate` ~1.5M times/sec over an
 * already-patched AVX2 vector hot loop (UTF-8->UTF-16 transcode shape). Each call
 * memcpys operands out of the regfile, runs the big vec_exec switch, memcpys the
 * result back; the enclosing asm thunk additionally spills+reloads all 16 ymm once
 * per run.
 *
 *   Path A (current): drive the REAL avxemu_emulate loop (== tramp_emulate_run)
 *                     over a regfile, one decoded[] entry per op. Optionally add a
 *                     representative per-run 16xymm spill+reload.
 *   Path B (Milestone-B ideal): the same run hand-written as straight-line SSE
 *                     intrinsics, each 256-bit op as 2x128-bit, all live vectors
 *                     kept in __m128i locals (operands loaded once, result stored
 *                     once -- register-resident, no per-op regfile traffic, no spill).
 *
 * Correctness gate runs first: Path B must be byte-identical to Path A (via the
 * real vec_exec) over random inputs, else timing is meaningless.
 *
 * Compile SSE-only exactly like the core (-msse4.2 -mno-avx -mno-fma): both the
 * emulator core AND Path B must run on the target CPU. NO AVX/AVX2 intrinsics.
 *
 * Build (standalone; does not touch build.sh's suite):
 *   see the build snippet at the bottom of this file / the spike report.
 */

#include "regfile.h"
#include "decode.h"
#include "vexops.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <emmintrin.h> /* SSE2  */
#include <tmmintrin.h> /* SSSE3 */
#include <smmintrin.h> /* SSE4.1 */
#include <nmmintrin.h> /* SSE4.2 */

#include <mach/mach_time.h>

/* ---- monotonic clock (Mavericks lacks clock_gettime CLOCK_MONOTONIC reliably) ---- */
static double now_sec(void) {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) mach_timebase_info(&tb);
    uint64_t t = mach_absolute_time();
    return (double)t * (double)tb.numer / (double)tb.denom / 1e9;
}

/* ============================================================================
 * The representative hot run.
 *
 * From the measured live op histogram (RULED-OUT.md): vpbroadcastd ~32%, plus
 * vpmovzxbw, vpsubb, vpand, vpor, vpcmpgtb -- a UTF-8->UTF-16 transcode shape.
 * A 9-op straight-line basic block with realistic data dependencies (outputs
 * feeding inputs), all 256-bit (VEX.L). vpbroadcastd appears 3/9 (~33%).
 *
 *   regs: IN0=ymm0  IN1=ymm1
 *         t0=ymm2 t1=ymm3 t2=ymm4 t3=ymm5 t4=ymm6 t5=ymm7 t6=ymm8 t7=ymm9 OUT=ymm10
 *
 *   1) t0  = vpbroadcastd(IN0)        ; broadcast dword0
 *   2) t1  = vpmovzxbw(IN0)           ; zero-extend low 16 bytes -> 16 words
 *   3) t2  = vpsubb(IN0, t0)          ; IN0 - t0
 *   4) t3  = vpbroadcastd(IN1)
 *   5) t4  = vpand(t2, t3)
 *   6) t5  = vpcmpgtb(t2, t0)         ; t2 > t0  (signed bytes)
 *   7) t6  = vpor(t4, t5)
 *   8) t7  = vpbroadcastd(t1)
 *   9) OUT = vpsubb(t6, t7)
 * ==========================================================================*/

#define RUN_N 9

enum { IN0=0, IN1=1, T0=2, T1=3, T2=4, T3=5, T4=6, T5=7, T6=8, T7=9, OUTR=10 };

/* Build one decoded vector-op entry (no memory operands; a=src1,b=src2). */
static decoded mk(vex_op op, int dst, int a_src, int b_src) {
    decoded d;
    memset(&d, 0, sizeof d);
    d.op       = op;
    d.type     = 0;
    d.len      = 5;                 /* arbitrary; no mem operand so unused for compute */
    d.a_src    = (int8_t)a_src;
    d.b_src    = (int8_t)b_src;
    d.c_src    = OPND_NONE;
    d.dst      = (int8_t)dst;
    d.dst_kind = DST_YMM;
    d.wide     = 1;                 /* 256-bit / VEX.L */
    d.imm      = 0;
    d.has_imm  = 0;
    d.has_mem  = 0;
    d.base     = OPND_NONE;
    d.index    = OPND_NONE;
    return d;
}

static void build_run(decoded run[RUN_N]) {
    run[0] = mk(VPBROADCASTD, T0,  OPND_NONE, IN0);
    run[1] = mk(VPMOVZXBW,    T1,  OPND_NONE, IN0);
    run[2] = mk(VPSUBB,       T2,  IN0,       T0);
    run[3] = mk(VPBROADCASTD, T3,  OPND_NONE, IN1);
    run[4] = mk(VPAND,        T4,  T2,        T3);
    run[5] = mk(VPCMPGTB,     T5,  T2,        T0);
    run[6] = mk(VPOR,         T6,  T4,        T5);
    run[7] = mk(VPBROADCASTD, T7,  OPND_NONE, T1);
    run[8] = mk(VPSUBB,       OUTR, T6,       T7);
}

/* ---- Path B: register-resident SSE, each 256-bit op as 2x128-bit ----
 * Operands loaded once from the regfile; all intermediates live in __m128i
 * locals (the compiler keeps them in xmm); result stored once. This is the
 * Milestone-B ideal for one run: NO per-op regfile memcpy, NO spill/reload. */
static inline void pathB_run(const avxemu_regfile *rf, avxemu_regfile *out) {
    /* load once */
    __m128i in0_lo = _mm_loadu_si128((const __m128i *)(rf->ymm[IN0]));
    __m128i in0_hi = _mm_loadu_si128((const __m128i *)(rf->ymm[IN0] + 16));
    __m128i in1_lo = _mm_loadu_si128((const __m128i *)(rf->ymm[IN1]));
    /* in1_hi unused by this run (broadcast only reads dword0 of the low half) */

    /* 1) t0 = vpbroadcastd(IN0): replicate dword0 across all 8 dwords */
    __m128i t0_lo = _mm_shuffle_epi32(in0_lo, 0x00);
    __m128i t0_hi = t0_lo;

    /* 2) t1 = vpmovzxbw(IN0): low 16 bytes -> 16 words */
    __m128i t1_lo = _mm_cvtepu8_epi16(in0_lo);
    __m128i t1_hi = _mm_cvtepu8_epi16(_mm_srli_si128(in0_lo, 8));

    /* 3) t2 = vpsubb(IN0, t0) */
    __m128i t2_lo = _mm_sub_epi8(in0_lo, t0_lo);
    __m128i t2_hi = _mm_sub_epi8(in0_hi, t0_hi);

    /* 4) t3 = vpbroadcastd(IN1) */
    __m128i t3_lo = _mm_shuffle_epi32(in1_lo, 0x00);
    __m128i t3_hi = t3_lo;

    /* 5) t4 = vpand(t2, t3) */
    __m128i t4_lo = _mm_and_si128(t2_lo, t3_lo);
    __m128i t4_hi = _mm_and_si128(t2_hi, t3_hi);

    /* 6) t5 = vpcmpgtb(t2, t0) */
    __m128i t5_lo = _mm_cmpgt_epi8(t2_lo, t0_lo);
    __m128i t5_hi = _mm_cmpgt_epi8(t2_hi, t0_hi);

    /* 7) t6 = vpor(t4, t5) */
    __m128i t6_lo = _mm_or_si128(t4_lo, t5_lo);
    __m128i t6_hi = _mm_or_si128(t4_hi, t5_hi);

    /* 8) t7 = vpbroadcastd(t1): dword0 of t1 (low half) */
    __m128i t7_lo = _mm_shuffle_epi32(t1_lo, 0x00);
    __m128i t7_hi = t7_lo;

    /* 9) OUT = vpsubb(t6, t7) */
    __m128i o_lo = _mm_sub_epi8(t6_lo, t7_lo);
    __m128i o_hi = _mm_sub_epi8(t6_hi, t7_hi);

    /* store once */
    _mm_storeu_si128((__m128i *)(out->ymm[OUTR]),      o_lo);
    _mm_storeu_si128((__m128i *)(out->ymm[OUTR] + 16), o_hi);
    (void)t1_hi; /* t1_hi computed for fidelity; this run's broadcast uses only t1_lo */
}

/* ---- Path A core: exactly what tramp_emulate_run does (set rip, emulate) ---- */
static void pathA_run(const decoded run[RUN_N], avxemu_regfile *rf) {
    for (int i = 0; i < RUN_N; i++) {
        rf->rip = run[i].dst;            /* any value; no mem operand consults it */
        if (!avxemu_emulate(&run[i], rf)) {
            fprintf(stderr, "avxemu_emulate failed on op %d\n", i);
            exit(2);
        }
    }
}

/* ---- representative per-run spill/reload: 16 ymm out + 16 ymm in.
 * The asm thunk does `vmovdqu ymmN` (256-bit). We are SSE-only, so model the
 * identical memory traffic as 32 x 128-bit stores + 32 x 128-bit loads
 * (16 ymm * 2 halves each direction = 512 bytes store + 512 bytes load). */
static uint8_t g_spillbuf[16 * 32] __attribute__((aligned(32)));
static inline uint64_t spill_reload(__m128i seed) {
    for (int i = 0; i < 32; i++)
        _mm_storeu_si128((__m128i *)(g_spillbuf + i * 16),
                         _mm_add_epi32(seed, _mm_set1_epi32(i)));
    __m128i acc = _mm_setzero_si128();
    for (int i = 0; i < 32; i++)
        acc = _mm_xor_si128(acc, _mm_loadu_si128((const __m128i *)(g_spillbuf + i * 16)));
    uint64_t lo; memcpy(&lo, &acc, 8);
    return lo;
}

/* ---- input slots (loop-variant, to defeat compiler hoisting of Path B) ---- */
#define NSLOT 1024
static ymm256 g_in[NSLOT];

static void init_inputs(unsigned seed) {
    /* deterministic xorshift fill */
    uint32_t s = seed ? seed : 0x9e3779b9u;
    for (int k = 0; k < NSLOT; k++)
        for (int j = 0; j < 32; j++) {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            g_in[k].b[j] = (uint8_t)s;
        }
}

static uint64_t reduce_out(const avxemu_regfile *rf) {
    uint64_t v[4]; memcpy(v, rf->ymm[OUTR], 32);
    return v[0] ^ v[1] ^ v[2] ^ v[3];
}

/* sink to keep results live */
static volatile uint64_t g_sink;

/* ============================================================================
 * Correctness gate
 * ==========================================================================*/
static int correctness_gate(const decoded run[RUN_N]) {
    int bad = 0;
    for (int trial = 0; trial < NSLOT; trial++) {
        avxemu_regfile rf; memset(&rf, 0, sizeof rf);
        memcpy(rf.ymm[IN0], g_in[trial].b, 32);
        memcpy(rf.ymm[IN1], g_in[(trial + 1) % NSLOT].b, 32);

        avxemu_regfile rb = rf;          /* same inputs */

        pathA_run(run, &rf);             /* real vec_exec path */
        pathB_run(&rb, &rb);             /* hand SSE */

        if (memcmp(rf.ymm[OUTR], rb.ymm[OUTR], 32) != 0) {
            if (bad < 4) {
                fprintf(stderr, "MISMATCH trial %d\n  A:", trial);
                for (int j = 0; j < 32; j++) fprintf(stderr, " %02x", rf.ymm[OUTR][j]);
                fprintf(stderr, "\n  B:");
                for (int j = 0; j < 32; j++) fprintf(stderr, " %02x", rb.ymm[OUTR][j]);
                fprintf(stderr, "\n");
            }
            bad++;
        }
    }
    return bad;
}

/* ============================================================================
 * Timed loops. Each "iteration" == one run (RUN_N op-executions).
 * ==========================================================================*/

/* Path A, dispatch only (the tramp_emulate_run loop). */
static double time_pathA_dispatch(const decoded run[RUN_N], long iters, uint64_t *chk) {
    avxemu_regfile rf; memset(&rf, 0, sizeof rf);
    uint64_t acc = 0;
    double t0 = now_sec();
    for (long i = 0; i < iters; i++) {
        int s = (int)(i & (NSLOT - 1));
        memcpy(rf.ymm[IN0], g_in[s].b, 32);
        memcpy(rf.ymm[IN1], g_in[(s + 1) & (NSLOT - 1)].b, 32);
        pathA_run(run, &rf);
        acc ^= reduce_out(&rf);
    }
    double t1 = now_sec();
    *chk = acc;
    return t1 - t0;
}

/* Path A + the per-run 16xymm spill/reload the live thunk also pays. */
static double time_pathA_full(const decoded run[RUN_N], long iters, uint64_t *chk) {
    avxemu_regfile rf; memset(&rf, 0, sizeof rf);
    uint64_t acc = 0;
    double t0 = now_sec();
    for (long i = 0; i < iters; i++) {
        int s = (int)(i & (NSLOT - 1));
        memcpy(rf.ymm[IN0], g_in[s].b, 32);
        memcpy(rf.ymm[IN1], g_in[(s + 1) & (NSLOT - 1)].b, 32);
        acc ^= spill_reload(_mm_set1_epi32((int)i));      /* per-run spill+reload */
        pathA_run(run, &rf);
        acc ^= reduce_out(&rf);
    }
    double t1 = now_sec();
    *chk = acc;
    return t1 - t0;
}

/* Path B, register-resident SSE. */
static double time_pathB(long iters, uint64_t *chk) {
    avxemu_regfile rf; memset(&rf, 0, sizeof rf);
    uint64_t acc = 0;
    double t0 = now_sec();
    for (long i = 0; i < iters; i++) {
        int s = (int)(i & (NSLOT - 1));
        memcpy(rf.ymm[IN0], g_in[s].b, 32);
        memcpy(rf.ymm[IN1], g_in[(s + 1) & (NSLOT - 1)].b, 32);
        pathB_run(&rf, &rf);
        acc ^= reduce_out(&rf);
    }
    double t1 = now_sec();
    *chk = acc;
    return t1 - t0;
}

/* Spill/reload alone (per-run), to report its share. */
static double time_spill(long iters, uint64_t *chk) {
    uint64_t acc = 0;
    double t0 = now_sec();
    for (long i = 0; i < iters; i++)
        acc ^= spill_reload(_mm_set1_epi32((int)i));
    double t1 = now_sec();
    *chk = acc;
    return t1 - t0;
}

int main(int argc, char **argv) {
    long iters = (argc > 1) ? atol(argv[1]) : 5000000L;   /* runs; *9 = op-executions */
    init_inputs(0xC0FFEEu);

    decoded run[RUN_N];
    build_run(run);

    /* ---- correctness gate FIRST ---- */
    int bad = correctness_gate(run);
    if (bad) {
        fprintf(stderr, "CORRECTNESS GATE FAILED: %d/%d trials mismatched. Aborting timing.\n",
                bad, NSLOT);
        return 1;
    }
    printf("correctness gate: PASS (%d/%d trials byte-identical, Path B == vec_exec)\n",
           NSLOT, NSLOT);

    /* ---- warm up ---- */
    uint64_t c;
    (void)time_pathA_dispatch(run, iters / 20 + 1, &c);
    (void)time_pathB(iters / 20 + 1, &c);

    /* ---- timed, best-of-3 (bimodal host; report min time = peak throughput) ---- */
    double a_disp = 1e30, a_full = 1e30, b = 1e30, sp = 1e30;
    uint64_t ca = 0, cf = 0, cb = 0, cs = 0;
    for (int rep = 0; rep < 3; rep++) {
        double x;
        x = time_pathA_dispatch(run, iters, &ca); if (x < a_disp) a_disp = x;
        x = time_pathA_full(run, iters, &cf);     if (x < a_full) a_full = x;
        x = time_pathB(iters, &cb);               if (x < b)      b = x;
        x = time_spill(iters, &cs);               if (x < sp)     sp = x;
    }
    g_sink = ca ^ cf ^ cb ^ cs;

    double ops = (double)iters * RUN_N;

    printf("\n--- timing (best of 3; iters=%ld runs, %.0f op-executions/path) ---\n",
           iters, ops);
    printf("Path A dispatch : %8.4f s  | %10.3f M ops/s | %9.3f M runs/s | %6.1f ns/run | %5.2f ns/op\n",
           a_disp, ops / a_disp / 1e6, iters / a_disp / 1e6,
           a_disp / iters * 1e9, a_disp / ops * 1e9);
    printf("Path A + spill  : %8.4f s  | %10.3f M ops/s | %9.3f M runs/s | %6.1f ns/run | %5.2f ns/op\n",
           a_full, ops / a_full / 1e6, iters / a_full / 1e6,
           a_full / iters * 1e9, a_full / ops * 1e9);
    printf("Path B (SSE)    : %8.4f s  | %10.3f M ops/s | %9.3f M runs/s | %6.1f ns/run | %5.2f ns/op\n",
           b, ops / b / 1e6, iters / b / 1e6,
           b / iters * 1e9, b / ops * 1e9);
    printf("spill/reload    : %8.4f s  | %38s | %6.1f ns/run\n",
           sp, "", sp / iters * 1e9);

    printf("\n--- speedup factors (Path B is the Milestone-B ideal) ---\n");
    printf("Path B vs Path A dispatch      : %6.2fx\n", a_disp / b);
    printf("Path B vs Path A + spill (real): %6.2fx\n", a_full / b);
    printf("spill share of Path A+spill    : %6.1f%%\n", 100.0 * sp / a_full);

    printf("\n--- projection vs live spin (1.5M avxemu_emulate/sec on Path A) ---\n");
    double live_emulate_per_s = 1.5e6;
    double live_runs_per_s = live_emulate_per_s / RUN_N;
    /* Path B per-run / Path A-dispatch per-run gives the wall-clock shrink of the
     * dispatch component; Path B eliminates spill entirely, so vs Path A+spill is
     * the fuller picture. */
    double f_disp = a_disp / b, f_full = a_full / b;
    printf("live Path A: %.2fM emulate/s => ~%.0fK runs/s\n",
           live_emulate_per_s / 1e6, live_runs_per_s / 1e3);
    printf("if hot-loop work is W seconds on Path A:\n");
    printf("  Path B (vs dispatch) => ~W/%.1f ; (vs dispatch+spill) => ~W/%.1f\n", f_disp, f_full);
    printf("(checksums A=%016llx A+sp=%016llx B=%016llx)\n",
           (unsigned long long)ca, (unsigned long long)cf, (unsigned long long)cb);
    return 0;
}
