/*
 * reloctest.c — round-trip test for avxemu_relocate_block (block-window
 * relocation, Task A).
 *
 * Each case builds a small executable buffer whose FIRST instruction is a
 * faulting BMI op and whose following instruction(s) are position-independent
 * legal ops, ending in `ret`. We:
 *   1. build the REFERENCE end-state: the dev host has no LZCNT/BMI, so native
 *      execution of the faulting op is NOT ground truth (lzcnt/tzcnt would run as
 *      bsr/bsf). Instead the reference runs the faulting op through the emulator
 *      (avxemu_emulate / bmi_exec) and the legal tail natively — exactly what the
 *      relocated block does. (The emulator's own correctness vs real BMI is
 *      validated separately on AVX2/BMI hardware by the bmi_oracle test.)
 *   2. call avxemu_relocate_block() on a second copy and assert it returns 1,
 *   3. run the patched copy,
 *   4. assert every GPR (except rsp) and the arithmetic flags are bit-identical.
 *
 * The relocated block runs the faulting op through avxemu_emulate (in the pool
 * stub), copies the legal ops verbatim, and jmps back — so a match proves the
 * relocation preserves machine state and control flow with no kernel trap.
 *
 * AVXEMU_FORCETRAMP is set so the BMI op is classified as faulting on this host.
 */
#include "decode.h"
#include "regfile.h"
#include "vexops.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/mman.h>

/* GPR x86 order (0=rax..7=rdi,8..15=r8..r15) at offset 0; rflags at offset 128. */
typedef struct { uint64_t gpr[16]; uint64_t rflags; } State;

/* Load `in` into the machine, call `code` (which ends in `ret`), capture into
 * `out`. rsp (gpr[4]) is the harness's own and is neither loaded nor captured. */
extern void reloc_test_invoke(void *code, const State *in, State *out);

__asm__(
"   .data\n"
"   .p2align 3\n"
"_rh_code: .quad 0\n"
"_rh_out:  .quad 0\n"
"_rh_rax:  .quad 0\n"
"   .text\n"
"   .p2align 4\n"
"   .globl _reloc_test_invoke\n"
"_reloc_test_invoke:\n"                 /* rdi=code, rsi=in, rdx=out */
"   pushq %rbx\n   pushq %rbp\n   pushq %r12\n   pushq %r13\n   pushq %r14\n   pushq %r15\n"
"   movq %rdi, _rh_code(%rip)\n"
"   movq %rdx, _rh_out(%rip)\n"
"   pushq 128(%rsi)\n   popfq\n"        /* rflags */
"   movq 0*8(%rsi), %rax\n"
"   movq 1*8(%rsi), %rcx\n"
"   movq 2*8(%rsi), %rdx\n"
"   movq 3*8(%rsi), %rbx\n"
"   movq 5*8(%rsi), %rbp\n"
"   movq 7*8(%rsi), %rdi\n"
"   movq 8*8(%rsi), %r8\n"
"   movq 9*8(%rsi), %r9\n"
"   movq 10*8(%rsi), %r10\n"
"   movq 11*8(%rsi), %r11\n"
"   movq 12*8(%rsi), %r12\n"
"   movq 13*8(%rsi), %r13\n"
"   movq 14*8(%rsi), %r14\n"
"   movq 15*8(%rsi), %r15\n"
"   movq 6*8(%rsi), %rsi\n"             /* rsi last */
"   callq *_rh_code(%rip)\n"            /* run the buffer (ends in ret) */
"   movq %rax, _rh_rax(%rip)\n"         /* stash rax (mov: no flag effect) */
"   movq _rh_out(%rip), %rax\n"
"   movq %rcx, 1*8(%rax)\n"
"   movq %rdx, 2*8(%rax)\n"
"   movq %rbx, 3*8(%rax)\n"
"   movq %rbp, 5*8(%rax)\n"
"   movq %rsi, 6*8(%rax)\n"
"   movq %rdi, 7*8(%rax)\n"
"   movq %r8,  8*8(%rax)\n"
"   movq %r9,  9*8(%rax)\n"
"   movq %r10, 10*8(%rax)\n"
"   movq %r11, 11*8(%rax)\n"
"   movq %r12, 12*8(%rax)\n"
"   movq %r13, 13*8(%rax)\n"
"   movq %r14, 14*8(%rax)\n"
"   movq %r15, 15*8(%rax)\n"
"   pushfq\n   popq %rcx\n   movq %rcx, 128(%rax)\n"   /* flags (still the buffer's) */
"   movq _rh_rax(%rip), %rcx\n   movq %rcx, 0*8(%rax)\n"
"   popq %r15\n   popq %r14\n   popq %r13\n   popq %r12\n   popq %rbp\n   popq %rbx\n"
"   ret\n"
);

#define ARITH_FLAGS 0x8D5ull   /* CF|PF|AF|ZF|SF|OF */
#define F_AF 0x10ull

/* Task B inline-lowering hook: incremented once per block that avxemu_relocate_block
 * emits via a native inline lowering (instead of the emulator stub). The BMI cases
 * below assert this advances, proving the INLINE path ran (not the stub). */
extern int avxemu_reloc_inlined;
extern int avxemu_reloc_last_reason;
extern void *avxemu_pool_base(void);
extern int avxemu_pool_init(void *hint, size_t cap);

/* Test buffers must sit within jmp-rel32 (+/-2GB) of the relocation pool, else
 * avxemu_relocate_block legitimately declines with reason 7 (rel32 range) and
 * the round-trip is never exercised. On 10.9 the pool is placed next to the
 * image and the default mmap(0,...) buffers land close enough by luck. On
 * macOS 15 the kernel routes the RWX pool to a far zone (~+7.6GB from a
 * low-hinted buffer) and refuses to co-locate RWX near the image, so every
 * case failed R7 there. Fix: pre-init the pool once, then mmap each code
 * buffer with a HINT just past the pool so both live in the same zone and
 * rel32 always reaches. Placement-only — it does not weaken what is tested
 * (the relocation/patch logic runs identically wherever the bytes sit). */
#define RT_POOL_CAP (96u << 20)
static uint8_t *rt_pool_hint_cursor;
static void rt_pool_ready(void) {
    if (avxemu_pool_base()) return;
    /* hint the pool near this function's own address (in-image on every host) */
    void *hint = (void *)(((uintptr_t)&rt_pool_ready + 0x100000) & ~(uintptr_t)0xfff);
    avxemu_pool_init(hint, RT_POOL_CAP);
    uint8_t *base = (uint8_t *)avxemu_pool_base();
    /* park buffers a few MB past the pool's end: same zone, no overlap */
    rt_pool_hint_cursor = base ? base + RT_POOL_CAP + (16u << 20) : 0;
}

/* Part 3 (Task C) test hook: register a synthetic function [base,base+size) as the
 * cached layout so avxemu_patch_safe runs against these mmap'd buffers (which are
 * not in the image's __text). avxemu_relocate_block calls avxemu_patch_safe(site,5)
 * before patching; without a registered region it would conservatively decline. */
extern void avxemu_patch_safe_test_region(uint8_t *base, size_t size);
extern int avxemu_patch_safe(uint8_t *site, int jmplen);

static int g_fail;

static void seed(State *s) {
    for (int i = 0; i < 16; i++)
        s->gpr[i] = 0x1111111100000000ull * (uint64_t)(i + 1) + 0xABCD + (uint64_t)i;
    s->rflags = 0x202;   /* IF + reserved */
}

static uint8_t *make_code(const uint8_t *bytes, size_t n) {
    rt_pool_ready();
    /* try a hint near the pool first (keeps rel32 in range on far-zone hosts);
     * fall back to any placement if the hint is unavailable/refused. */
    uint8_t *p = MAP_FAILED;
    if (rt_pool_hint_cursor) {
        p = mmap(rt_pool_hint_cursor, 0x1000, PROT_READ | PROT_WRITE | PROT_EXEC,
                 MAP_PRIVATE | MAP_ANON, -1, 0);
        if (p != MAP_FAILED) rt_pool_hint_cursor += 0x1000;
    }
    if (p == MAP_FAILED)
        p = mmap(0, 0x1000, PROT_READ | PROT_WRITE | PROT_EXEC,
                 MAP_PRIVATE | MAP_ANON, -1, 0);
    if (p == MAP_FAILED) return 0;
    memset(p, 0xCC, 0x1000);
    memcpy(p, bytes, n);
    return p;
}

static void run_case(const char *name, const uint8_t *bytes, size_t n) {
    uint8_t *test_buf = make_code(bytes, n);
    if (!test_buf) { printf("  [%s] mmap failed\n", name); g_fail++; return; }
    avxemu_patch_safe_test_region(test_buf, n);   /* Part 3: register this buffer's bounds */

    State in, ref, got;
    seed(&in);

    /* (1) reference end-state: emulate the faulting op through avxemu_emulate
     * (the dev host is itself a no-LZCNT/BMI target, so native execution of the
     * faulting op is NOT ground truth — it runs as bsr/bsf), then run the legal
     * tail (which IS universally supported) natively. This mirrors exactly what
     * the relocated block does. */
    decoded fd;
    int fl = decode(bytes, &fd);
    if (fl <= 0) { printf("  [%s] decode failed\n", name); g_fail++; return; }
    avxemu_regfile rf;
    memset(&rf, 0, sizeof rf);
    for (int i = 0; i < 16; i++) rf.gpr[i] = in.gpr[i];
    rf.rflags = in.rflags;
    rf.rip = (uint64_t)(uintptr_t)test_buf;
    if (!avxemu_emulate(&fd, &rf)) { printf("  [%s] emulate failed\n", name); g_fail++; return; }
    State mid;
    for (int i = 0; i < 16; i++) mid.gpr[i] = rf.gpr[i];
    mid.rflags = rf.rflags;
    uint8_t *legal_buf = make_code(bytes + fl, n - fl);   /* legal tail + ret */
    if (!legal_buf) { printf("  [%s] mmap failed\n", name); g_fail++; return; }
    memset(&ref, 0, sizeof ref);
    reloc_test_invoke(legal_buf, &mid, &ref);

    /* (2) relocate the buffer */
    int r = avxemu_relocate_block(test_buf);
    if (r != 1) { printf("  [%s] relocate returned %d (expected 1, reason %d, site=%p pool=%p) — FAIL\n", name, r, avxemu_reloc_last_reason, (void *)test_buf, avxemu_pool_base()); g_fail++; return; }

    /* (3) run the patched buffer */
    memset(&got, 0, sizeof got);
    reloc_test_invoke(test_buf, &in, &got);

    /* (4) compare GPRs (skip 4=rsp) and arithmetic flags */
    int bad = 0;
    for (int i = 0; i < 16; i++) {
        if (i == 4) continue;
        if (ref.gpr[i] != got.gpr[i]) {
            printf("  [%s] gpr%d MISMATCH ref=%016llx got=%016llx\n", name, i,
                   (unsigned long long)ref.gpr[i], (unsigned long long)got.gpr[i]);
            bad = 1;
        }
    }
    if ((ref.rflags & ARITH_FLAGS) != (got.rflags & ARITH_FLAGS)) {
        printf("  [%s] flags MISMATCH ref=%03llx got=%03llx\n", name,
               (unsigned long long)(ref.rflags & ARITH_FLAGS),
               (unsigned long long)(got.rflags & ARITH_FLAGS));
        bad = 1;
    }
    g_fail += bad;
    printf("  %-28s %s\n", name, bad ? "FAIL" : "ok");
}

/* Regression: a window whose FOLLOWER is a position-dependent relative control
 * transfer (call rel32 / loop rel8) must be DECLINED — avxemu_relocate_block
 * returns 0 so the caller keeps using SIGILL emulation. (Copying such a follower
 * verbatim would recompute its rel target from the pool address: wrong callee /
 * wrong loop target, and call would push a pool-relative return address.) */
static void run_decline(const char *name, const uint8_t *bytes, size_t n) {
    uint8_t *test_buf = make_code(bytes, n);
    if (!test_buf) { printf("  [%s] mmap failed\n", name); g_fail++; return; }
    avxemu_patch_safe_test_region(test_buf, n);   /* Part 3: register this buffer's bounds */
    int r = avxemu_relocate_block(test_buf);
    if (r != 0) {
        printf("  [%s] relocate returned %d (expected 0/declined) — FAIL\n", name, r);
        g_fail++;
        return;
    }
    printf("  %-28s %s\n", name, "ok (declined)");
}

/* Per-op BMI lowering case: build a buffer whose faulting op is `bytes`, set the
 * source register(s) to specific inputs, then verify the relocated (INLINE-lowered)
 * block reproduces bmi_exec's destination AND arithmetic flags bit-for-bit.
 *
 * The reference is avxemu_emulate (bmi_exec) on the faulting op followed by the
 * legal tail natively — identical to what the relocated block must do, but with
 * the faulting op now lowered to a native sequence rather than a C call. We assert
 * avxemu_reloc_inlined advanced so the test proves the inline path, not the stub.
 * `s2reg < 0` => single-source op (lzcnt/tzcnt). */
static void run_bmi_case(const char *name, const uint8_t *bytes, size_t n,
                         vex_op want_op, int want_opsize,
                         int s1reg, uint64_t s1val,
                         int s2reg, uint64_t s2val,
                         uint64_t seed_flags) {
    uint8_t *test_buf = make_code(bytes, n);
    if (!test_buf) { printf("  [%s] mmap failed\n", name); g_fail++; return; }
    avxemu_patch_safe_test_region(test_buf, n);   /* Part 3: register this buffer's bounds */

    decoded fd;
    int fl = decode(bytes, &fd);
    if (fl <= 0) { printf("  [%s] decode failed\n", name); g_fail++; return; }
    if (fd.op != want_op || fd.opsize != want_opsize) {
        printf("  [%s] decode mismatch: op=%d opsize=%d (want %d/%d) — FAIL\n",
               name, (int)fd.op, fd.opsize, (int)want_op, want_opsize);
        g_fail++; return;
    }

    State in; seed(&in);
    in.rflags = seed_flags;
    if (s1reg >= 0) in.gpr[s1reg] = s1val;
    if (s2reg >= 0) in.gpr[s2reg] = s2val;

    /* reference end-state (emulate faulting op + native tail) */
    avxemu_regfile rf;
    memset(&rf, 0, sizeof rf);
    for (int i = 0; i < 16; i++) rf.gpr[i] = in.gpr[i];
    rf.rflags = in.rflags;
    rf.rip = (uint64_t)(uintptr_t)test_buf;
    if (!avxemu_emulate(&fd, &rf)) { printf("  [%s] emulate failed\n", name); g_fail++; return; }
    State mid;
    for (int i = 0; i < 16; i++) mid.gpr[i] = rf.gpr[i];
    mid.rflags = rf.rflags;
    uint8_t *legal_buf = make_code(bytes + fl, n - fl);
    if (!legal_buf) { printf("  [%s] mmap failed\n", name); g_fail++; return; }
    State ref;
    memset(&ref, 0, sizeof ref);
    reloc_test_invoke(legal_buf, &mid, &ref);

    /* relocate, asserting the INLINE lowering was used (counter advances by 1) */
    int before = avxemu_reloc_inlined;
    int r = avxemu_relocate_block(test_buf);
    if (r != 1) { printf("  [%s] relocate returned %d (expected 1, reason %d, site=%p pool=%p) — FAIL\n", name, r, avxemu_reloc_last_reason, (void *)test_buf, avxemu_pool_base()); g_fail++; return; }
    if (avxemu_reloc_inlined != before + 1) {
        printf("  [%s] inline lowering NOT used (stub fallback) — FAIL\n", name);
        g_fail++; return;
    }

    State got;
    memset(&got, 0, sizeof got);
    reloc_test_invoke(test_buf, &in, &got);

    int bad = 0;
    for (int i = 0; i < 16; i++) {
        if (i == 4) continue;
        if (ref.gpr[i] != got.gpr[i]) {
            printf("  [%s] gpr%d MISMATCH ref=%016llx got=%016llx\n", name, i,
                   (unsigned long long)ref.gpr[i], (unsigned long long)got.gpr[i]);
            bad = 1;
        }
    }
    if ((ref.rflags & ARITH_FLAGS) != (got.rflags & ARITH_FLAGS)) {
        printf("  [%s] flags MISMATCH ref=%03llx got=%03llx\n", name,
               (unsigned long long)(ref.rflags & ARITH_FLAGS),
               (unsigned long long)(got.rflags & ARITH_FLAGS));
        bad = 1;
    }
    g_fail += bad;
    printf("  %-40s %s\n", name, bad ? "FAIL" : "ok");
}

/* Drive one (op,opsize) template over a spread of inputs (incl. 0, all-ones, 1),
 * under both PF/AF=0 and PF/AF=1 seedings to prove PF/AF are PRESERVED. */
static void drive_1src(const char *tag, const uint8_t *bytes, size_t n,
                       vex_op op, int opsize, int srcreg) {
    static const uint64_t v32[] = { 0, 1, 0xFFFFFFFFu, 0x80000000u, 0x00010000u,
                                    0x0F0F0F0Fu, 0xDEADBEEFu, 0x00000010u };
    static const uint64_t v64[] = { 0, 1, ~0ull, 0x8000000000000000ull, 0x100000000ull,
                                    0x0F0F0F0F0F0F0F0Full, 0xDEADBEEFCAFEBABEull, 0x40ull };
    const uint64_t *vs = (opsize == 64) ? v64 : v32;
    int nv = 8;
    const uint64_t seeds[2] = { 0x202ull, 0x202ull | ARITH_FLAGS };
    for (int sfi = 0; sfi < 2; sfi++)
        for (int i = 0; i < nv; i++) {
            char nm[96];
            uint64_t sv = vs[i];
            if (opsize == 32) sv |= 0xCAFEF00D00000000ull; /* high garbage: must be masked off */
            snprintf(nm, sizeof nm, "%s in=%016llx f=%llx", tag,
                     (unsigned long long)sv, (unsigned long long)seeds[sfi]);
            run_bmi_case(nm, bytes, n, op, opsize, srcreg, sv, -1, 0, seeds[sfi]);
        }
}

static void drive_2src(const char *tag, const uint8_t *bytes, size_t n,
                       vex_op op, int opsize, int s1reg, int s2reg) {
    static const uint64_t a[] = { 0, ~0ull, 1, 0x0F0F0F0F0F0F0F0Full, 0x8000000000000000ull,
                                  0xDEADBEEFull, 0xFFFFull, 0x40ull };
    static const uint64_t b[] = { 0, ~0ull, 0xFFFFFFFFull, 0xF0F0F0F0F0F0F0F0ull, 1,
                                  0xCAFEBABEull, 0x21ull, 0x80ull };
    int nv = 8;
    const uint64_t seeds[2] = { 0x202ull, 0x202ull | ARITH_FLAGS };
    for (int sfi = 0; sfi < 2; sfi++)
        for (int i = 0; i < nv; i++) {
            uint64_t s1 = a[i], s2 = b[i];
            if (opsize == 32) { s1 |= 0x1111111100000000ull; s2 |= 0x2222222200000000ull; }
            char nm[112];
            snprintf(nm, sizeof nm, "%s s1=%016llx s2=%016llx f=%llx", tag,
                     (unsigned long long)s1, (unsigned long long)s2,
                     (unsigned long long)seeds[sfi]);
            run_bmi_case(nm, bytes, n, op, opsize, s1reg, s1, s2reg, s2, seeds[sfi]);
        }
}

/* Fallback proof: a BMI op NOT in the inline table (e.g. blsi) must still
 * relocate (return 1) via the emulator stub, leaving avxemu_reloc_inlined
 * UNCHANGED, and reproduce bmi_exec's result. */
static void run_stub_case(const char *name, const uint8_t *bytes, size_t n,
                          int srcreg, uint64_t srcval) {
    uint8_t *test_buf = make_code(bytes, n);
    if (!test_buf) { printf("  [%s] mmap failed\n", name); g_fail++; return; }
    avxemu_patch_safe_test_region(test_buf, n);   /* Part 3: register this buffer's bounds */
    decoded fd;
    int fl = decode(bytes, &fd);
    if (fl <= 0) { printf("  [%s] decode failed\n", name); g_fail++; return; }
    State in; seed(&in);
    if (srcreg >= 0) in.gpr[srcreg] = srcval;
    avxemu_regfile rf; memset(&rf, 0, sizeof rf);
    for (int i = 0; i < 16; i++) rf.gpr[i] = in.gpr[i];
    rf.rflags = in.rflags; rf.rip = (uint64_t)(uintptr_t)test_buf;
    if (!avxemu_emulate(&fd, &rf)) { printf("  [%s] emulate failed\n", name); g_fail++; return; }
    State mid; for (int i = 0; i < 16; i++) mid.gpr[i] = rf.gpr[i]; mid.rflags = rf.rflags;
    uint8_t *legal_buf = make_code(bytes + fl, n - fl);
    State ref; memset(&ref, 0, sizeof ref);
    reloc_test_invoke(legal_buf, &mid, &ref);
    int before = avxemu_reloc_inlined;
    int r = avxemu_relocate_block(test_buf);
    if (r != 1) { printf("  [%s] relocate returned %d (expected 1, reason %d, site=%p pool=%p) — FAIL\n", name, r, avxemu_reloc_last_reason, (void *)test_buf, avxemu_pool_base()); g_fail++; return; }
    if (avxemu_reloc_inlined != before) {
        printf("  [%s] expected STUB but inline counter advanced — FAIL\n", name); g_fail++; return;
    }
    State got; memset(&got, 0, sizeof got);
    reloc_test_invoke(test_buf, &in, &got);
    int bad = 0;
    for (int i = 0; i < 16; i++) { if (i == 4) continue;
        if (ref.gpr[i] != got.gpr[i]) { printf("  [%s] gpr%d MISMATCH\n", name, i); bad = 1; } }
    if ((ref.rflags & ARITH_FLAGS) != (got.rflags & ARITH_FLAGS)) { printf("  [%s] flags MISMATCH\n", name); bad = 1; }
    g_fail += bad;
    printf("  %-40s %s\n", name, bad ? "FAIL" : "ok (stub)");
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setenv("AVXEMU_FORCETRAMP", "1", 1);   /* classify the BMI op as faulting on any host */
    printf("== block-window relocation: emulate + verbatim + jmp-back ==\n");

    /* Case 1: faulting op (4B) + ONE legal op (2B) -> window reaches 6 (overshoot).
     *   lzcnt eax,edi ; neg edx ; ret
     * window = [lzcnt, neg edx); jmp-back lands on `ret`. */
    {
        static const uint8_t code[] = {
            0xF3, 0x0F, 0xBD, 0xC7,   /* lzcnt eax, edi   (4) */
            0xF7, 0xDA,               /* neg   edx        (2) */
            0xC3                      /* ret */
        };
        run_case("lzcnt;neg edx (window=6)", code, sizeof code);
    }

    /* Case 2: faulting op (4B) + a 1-byte legal op -> window reaches exactly 5,
     * with a further legal op executed natively after the jmp-back.
     *   tzcnt eax,edi ; cltd ; neg edx ; ret
     * window = [tzcnt, cltd); jmp-back lands on `neg edx`.
     * (A window holding two legal instructions is unreachable: the shortest
     *  faulting BMI op is 4 bytes, so a single following instruction always
     *  reaches the 5-byte threshold. This case instead exercises the exact-5
     *  boundary plus verbatim copy of a distinct 1-byte op.) */
    {
        static const uint8_t code[] = {
            0xF3, 0x0F, 0xBC, 0xC7,   /* tzcnt eax, edi   (4) */
            0x99,                     /* cltd             (1) */
            0xF7, 0xDA,               /* neg   edx        (2) */
            0xC3                      /* ret */
        };
        run_case("tzcnt;cltd (window=5)", code, sizeof code);
    }

    /* Regression case 3: faulting op (4B) + follower = `call rel32` (E8) must be
     * DECLINED (position-dependent rel target + pool-relative return address). */
    {
        static const uint8_t code[] = {
            0xF3, 0x0F, 0xBD, 0xC7,         /* lzcnt eax, edi    (4) */
            0xE8, 0x00, 0x00, 0x00, 0x00,   /* call rel32        (5) */
            0xC3                            /* ret */
        };
        run_decline("lzcnt;call rel32 (decline)", code, sizeof code);
    }

    /* Regression case 4: faulting op (4B) + follower = `loop rel8` (E2) must be
     * DECLINED (position-dependent loop target). */
    {
        static const uint8_t code[] = {
            0xF3, 0x0F, 0xBD, 0xC7,   /* lzcnt eax, edi   (4) */
            0xE2, 0xFE,               /* loop  .          (2) */
            0xC3                      /* ret */
        };
        run_decline("lzcnt;loop rel8 (decline)", code, sizeof code);
    }

    /* Part 3 (Task C) window patch-safety: an inbound branch elsewhere in the
     * function targets a byte INSIDE the 5-byte jmp footprint (site, site+5).
     * Patching would corrupt that entry path, so avxemu_patch_safe must make
     * avxemu_relocate_block decline (return 0).
     *   0: lzcnt eax,edi   (4B faulting site; jmp footprint = [0,5))
     *   4: nop             (1B follower; offset 4 is INSIDE the open interval (0,5))
     *   5: jmp -3 -> 4     (a branch whose target is offset 4)
     *   7: ret
     * Contrast with case 1 (lzcnt;neg), an identical-shape window with NO such
     * inbound branch, which relocates successfully. */
    {
        static const uint8_t code[] = {
            0xF3, 0x0F, 0xBD, 0xC7,   /* lzcnt eax, edi   (4) */
            0x90,                     /* nop              (1) */
            0xEB, 0xFD,               /* jmp .-3  -> off 4 (inside the jmp footprint) */
            0xC3                      /* ret */
        };
        run_decline("lzcnt;branch into footprint (decline)", code, sizeof code);
    }

    /* ---- Jump-table-aware patch safety (fault-storm fix) --------------------
     * The dominant repeat-faulting sites live in functions whose only indirect
     * jmp is the bounded LLVM jump-table dispatch:
     *     cmp rI,imm ; ja default ; lea rB,[rip+tbl] ; movsxd rD,[rB+rI*4] ;
     *     add rD,rB ; jmp *rD          (table = rel32 entries from tbl base)
     * collect_branch_targets must RESOLVE that dispatch — enumerate the table's
     * targets and skip the inline table data — instead of blanket-declining on
     * has_indirect. Layout of the synthetic function (offsets):
     *   0: lzcnt edi,edi   (4B faulting site; jmp footprint [0,5))
     *   4: add  edi,-28    (copyable follower -> window = 7 bytes)
     *   7: cmp  edi,3
     *  10: ja   +16 -> 28  (default, past the dispatch)
     *  12: lea  rax,[rip+13] -> table at 32
     *  19: movsxd rdi,[rax+rdi*4]
     *  23: add  rdi,rax
     *  26: jmp *rdi
     *  28: ret            (default + all table targets)
     *  29: int3 pad
     *  32: table[4] = rel32 entries (28-32 = -4 each)                     */
    {
        static const uint8_t code[] = {
            0xF3, 0x0F, 0xBD, 0xFF,                    /*  0: lzcnt edi,edi */
            0x83, 0xC7, 0xE4,                          /*  4: add edi,-28 */
            0x83, 0xFF, 0x03,                          /*  7: cmp edi,3 */
            0x77, 0x10,                                /* 10: ja +16 -> 28 */
            0x48, 0x8D, 0x05, 0x0D, 0x00, 0x00, 0x00,  /* 12: lea rax,[rip+13] -> 32 */
            0x48, 0x63, 0x3C, 0xB8,                    /* 19: movsxd rdi,[rax+rdi*4] */
            0x48, 0x01, 0xC7,                          /* 23: add rdi,rax */
            0xFF, 0xE7,                                /* 26: jmp *rdi */
            0xC3,                                      /* 28: ret */
            0xCC, 0xCC, 0xCC,                          /* 29: pad */
            0xFC, 0xFF, 0xFF, 0xFF,                    /* 32: tbl[0] -> 28 */
            0xFC, 0xFF, 0xFF, 0xFF,                    /* 36: tbl[1] -> 28 */
            0xFC, 0xFF, 0xFF, 0xFF,                    /* 40: tbl[2] -> 28 */
            0xFC, 0xFF, 0xFF, 0xFF,                    /* 44: tbl[3] -> 28 */
        };
        run_case("lzcnt in jump-table fn (ok)", code, sizeof code);
    }

    /* Same function, but table entry 2 targets offset 2 — INSIDE the jmp
     * footprint (0,5). The resolver must mark it and patch_safe must decline. */
    {
        static const uint8_t code[] = {
            0xF3, 0x0F, 0xBD, 0xFF,                    /*  0: lzcnt edi,edi */
            0x83, 0xC7, 0xE4,                          /*  4: add edi,-28 */
            0x83, 0xFF, 0x03,                          /*  7: cmp edi,3 */
            0x77, 0x10,                                /* 10: ja +16 -> 28 */
            0x48, 0x8D, 0x05, 0x0D, 0x00, 0x00, 0x00,  /* 12: lea rax,[rip+13] -> 32 */
            0x48, 0x63, 0x3C, 0xB8,                    /* 19: movsxd rdi,[rax+rdi*4] */
            0x48, 0x01, 0xC7,                          /* 23: add rdi,rax */
            0xFF, 0xE7,                                /* 26: jmp *rdi */
            0xC3,                                      /* 28: ret */
            0xCC, 0xCC, 0xCC,                          /* 29: pad */
            0xFC, 0xFF, 0xFF, 0xFF,                    /* 32: tbl[0] -> 28 */
            0xFC, 0xFF, 0xFF, 0xFF,                    /* 36: tbl[1] -> 28 */
            0xE2, 0xFF, 0xFF, 0xFF,                    /* 40: tbl[2] -> 2 (in footprint!) */
            0xFC, 0xFF, 0xFF, 0xFF,                    /* 44: tbl[3] -> 28 */
        };
        run_decline("jump-table entry into footprint (decline)", code, sizeof code);
    }

    /* An UNRESOLVABLE indirect jmp (no lea/movsxd table pattern) must still
     * blanket-decline: has_indirect stays set, patch_safe returns 0. */
    {
        static const uint8_t code[] = {
            0xF3, 0x0F, 0xBD, 0xFF,   /* 0: lzcnt edi,edi */
            0x83, 0xC7, 0xE4,         /* 4: add edi,-28 (window ok) */
            0x48, 0x89, 0xC7,         /* 7: mov rdi,rax */
            0xFF, 0xE7,               /* 10: jmp *rdi (unresolvable) */
            0xC3,                     /* 12: ret */
        };
        run_decline("unresolvable indirect (decline)", code, sizeof code);
    }

    /* ---- Task B: inline native lowerings, oracle-gated vs bmi_exec ----------
     * Each template's faulting op is relocated and must run through the INLINE
     * native sequence (asserted via avxemu_reloc_inlined), matching bmi_exec's
     * destination AND CF/PF/AF/ZF/SF/OF for the same inputs, at opsize 32 + 64,
     * over inputs including 0 (zero-input lzcnt/tzcnt => result=opsize, CF=1),
     * all-ones, 1, and assorted values, under PF/AF=0 and PF/AF=1 seedings. */
    printf("\n== inline BMI lowerings: native sequence vs bmi_exec (dst + flags) ==\n");

    /* TZCNT eax,edi (4B) + nop -> window 5 ; rax<-result, src=edi(7) */
    { static const uint8_t c[] = { 0xF3,0x0F,0xBC,0xC7, 0x90, 0xC3 };
      drive_1src("tzcnt32 eax,edi", c, sizeof c, BMI_TZCNT, 32, 7); }
    /* TZCNT rax,rdi (5B, self-contained window) */
    { static const uint8_t c[] = { 0xF3,0x48,0x0F,0xBC,0xC7, 0xC3 };
      drive_1src("tzcnt64 rax,rdi", c, sizeof c, BMI_TZCNT, 64, 7); }
    /* LZCNT eax,edi (4B) + nop */
    { static const uint8_t c[] = { 0xF3,0x0F,0xBD,0xC7, 0x90, 0xC3 };
      drive_1src("lzcnt32 eax,edi", c, sizeof c, BMI_LZCNT, 32, 7); }
    /* LZCNT rax,rdi (5B) */
    { static const uint8_t c[] = { 0xF3,0x48,0x0F,0xBD,0xC7, 0xC3 };
      drive_1src("lzcnt64 rax,rdi", c, sizeof c, BMI_LZCNT, 64, 7); }

    /* ANDN eax,ecx,edx  (eax = ~ecx & edx) : src1=ecx(1), src2=edx(2) */
    { static const uint8_t c[] = { 0xC4,0xE2,0x70,0xF2,0xC2, 0xC3 };
      drive_2src("andn32 eax,ecx,edx", c, sizeof c, BMI_ANDN, 32, 1, 2); }
    /* ANDN rax,rcx,rdx */
    { static const uint8_t c[] = { 0xC4,0xE2,0xF0,0xF2,0xC2, 0xC3 };
      drive_2src("andn64 rax,rcx,rdx", c, sizeof c, BMI_ANDN, 64, 1, 2); }

    /* SHLX eax,edx,ecx  (eax = edx << ecx) : val=edx(2), count=ecx(1) */
    { static const uint8_t c[] = { 0xC4,0xE2,0x71,0xF7,0xC2, 0xC3 };
      drive_2src("shlx32 eax,edx,ecx", c, sizeof c, BMI_SHLX, 32, 2, 1); }
    /* SHLX rax,rdx,rcx */
    { static const uint8_t c[] = { 0xC4,0xE2,0xF1,0xF7,0xC2, 0xC3 };
      drive_2src("shlx64 rax,rdx,rcx", c, sizeof c, BMI_SHLX, 64, 2, 1); }

    /* ---- Fix 1 regression: dst==src for tzcnt/lzcnt -----------------------
     * The original emitter captured CF=(src==0) with `test S,S; setz` AFTER
     * bsf/bsr had already overwritten S when D==S (e.g. tzcnt eax,eax), so CF
     * came out as (result==0) instead of (src==0). These cases (dst==src, both
     * opsizes, inputs incl. 0/1/odd/even, PF/AF=0 and =1) lock the fix in and
     * assert dst + CF/ZF/SF/OF vs bmi_exec with PF/AF preserved. */
    printf("\n== Fix 1 regression: tzcnt/lzcnt dst==src (CF capture before bsf/bsr) ==\n");
    /* tzcnt eax,eax (F3 0F BC C0) + nop ; src reg = rax(0) */
    { static const uint8_t c[] = { 0xF3,0x0F,0xBC,0xC0, 0x90, 0xC3 };
      drive_1src("tzcnt32 eax,eax (dst==src)", c, sizeof c, BMI_TZCNT, 32, 0); }
    /* tzcnt rax,rax (F3 48 0F BC C0) */
    { static const uint8_t c[] = { 0xF3,0x48,0x0F,0xBC,0xC0, 0xC3 };
      drive_1src("tzcnt64 rax,rax (dst==src)", c, sizeof c, BMI_TZCNT, 64, 0); }
    /* lzcnt eax,eax (F3 0F BD C0) + nop */
    { static const uint8_t c[] = { 0xF3,0x0F,0xBD,0xC0, 0x90, 0xC3 };
      drive_1src("lzcnt32 eax,eax (dst==src)", c, sizeof c, BMI_LZCNT, 32, 0); }
    /* lzcnt rax,rax (F3 48 0F BD C0) */
    { static const uint8_t c[] = { 0xF3,0x48,0x0F,0xBD,0xC0, 0xC3 };
      drive_1src("lzcnt64 rax,rax (dst==src)", c, sizeof c, BMI_LZCNT, 64, 0); }

    /* ---- SHLX rcx-aliasing coverage (exercises the dst==rcx drop8 path and
     * the push-V/C-before-rcx-change ordering) at opsize 32 and 64 ---------- */
    printf("\n== SHLX rcx aliasing: D==C, V==C, D==V ==\n");
    /* D==C: shlx ecx,edx,ecx  dst=rcx(1) val=edx(2) cnt=rcx(1)  (ModRM C0|1<<3|2=CA) */
    { static const uint8_t c[] = { 0xC4,0xE2,0x71,0xF7,0xCA, 0xC3 };
      drive_2src("shlx32 ecx,edx,ecx (D==C)", c, sizeof c, BMI_SHLX, 32, 2, 1); }
    { static const uint8_t c[] = { 0xC4,0xE2,0xF1,0xF7,0xCA, 0xC3 };
      drive_2src("shlx64 rcx,rdx,rcx (D==C)", c, sizeof c, BMI_SHLX, 64, 2, 1); }
    /* V==C: shlx eax,ecx,ecx  dst=rax(0) val=rcx(1) cnt=rcx(1)  (ModRM C0|0|1=C1) */
    { static const uint8_t c[] = { 0xC4,0xE2,0x71,0xF7,0xC1, 0xC3 };
      drive_2src("shlx32 eax,ecx,ecx (V==C)", c, sizeof c, BMI_SHLX, 32, 1, 1); }
    { static const uint8_t c[] = { 0xC4,0xE2,0xF1,0xF7,0xC1, 0xC3 };
      drive_2src("shlx64 rax,rcx,rcx (V==C)", c, sizeof c, BMI_SHLX, 64, 1, 1); }
    /* D==V: shlx ecx,ecx,edx  dst=rcx(1) val=rcx(1) cnt=rdx(2)  (vvvv=~2; ModRM C9) */
    { static const uint8_t c[] = { 0xC4,0xE2,0x69,0xF7,0xC9, 0xC3 };
      drive_2src("shlx32 ecx,ecx,edx (D==V)", c, sizeof c, BMI_SHLX, 32, 1, 2); }
    { static const uint8_t c[] = { 0xC4,0xE2,0xE9,0xF7,0xC9, 0xC3 };
      drive_2src("shlx64 rcx,rcx,rdx (D==V)", c, sizeof c, BMI_SHLX, 64, 1, 2); }

    /* Fallback: BLSI eax,edi (VEX.0F38.W0 F3 /3) is BMI but not inline-lowered;
     * it must relocate via the STUB (counter unchanged) and stay correct. */
    { static const uint8_t c[] = { 0xC4,0xE2,0x78,0xF3,0xDF, 0xC3 };
      run_stub_case("blsi32 eax,edi (stub fallback)", c, sizeof c, 7, 0x00F0); }

    /* patch_safe static-map cap: avxemu_patch_safe now backs its branch-target
     * scan with a fixed static buffer (PATCHSAFE_MAP_MAX = 256 KiB) instead of
     * malloc/free, so it is async-signal-safe inside the SIGILL handler. A
     * containing function whose span exceeds that cap must DECLINE (return 0, the
     * safe floor) rather than overrun the buffer. Register a synthetic function
     * span larger than the cap and assert the decline. The cap check returns
     * before any byte of the (here-unbacked) region is read, so no large buffer is
     * needed. */
    printf("\n== patch_safe static-map cap: oversized function span declines ==\n");
    {
        static uint8_t cap_base[64];                 /* only the base address matters */
        size_t huge = (256u * 1024) + 4096;          /* span > PATCHSAFE_MAP_MAX */
        avxemu_patch_safe_test_region(cap_base, huge);
        int r = avxemu_patch_safe(cap_base, 5);
        if (r != 0) { printf("  oversized span: patch_safe returned %d (expected 0) — FAIL\n", r); g_fail++; }
        else        printf("  %-28s %s\n", "oversized function span", "ok (declined)");
    }

    printf("\nRELOCTEST TOTAL: %d failure(s)\n", g_fail);
    return g_fail ? 1 : 0;
}
