/*
 * zcnt16.c — regression test for 16-bit (66-prefixed) lzcnt/tzcnt decode+emulate.
 *
 * THE bug this pins down: decode.c ignored the 66 operand-size prefix on
 * F3 0F BD/BC, so `lzcnt cx,di` (66 f3 0f bd cf — the hot instruction in
 * Claude Code's JSC 16-bit string search) was emulated as 32-bit. With the
 * upper half of the source zero (as after vpmovmskb), lzcnt32 = 16 + lzcnt16:
 * every result was off by +16, the app's character-index math went wrong, and
 * its search loop never terminated -> the "no-AVX2 startup spin". Native CPUs
 * ran the true 16-bit op and finished instantly, which is why the hang was
 * only ever seen under emulation.
 *
 * Ground truth here is hand-computed (lzcnt/tzcnt of small constants), so the
 * test is hermetic and runs on any x86-64 host, no BMI hardware needed.
 */
#include "decode.h"
#include "vexops.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define FLAG_CF 0x0001ull
#define FLAG_ZF 0x0040ull

static int fails = 0;

static void chk(const char *name, const uint8_t *bytes, int nbytes,
                int want_op, int want_opsize, int want_len,
                int want_dst, int want_src,
                uint64_t src_val, uint64_t want_result,
                int want_cf, int want_zf) {
    decoded d; memset(&d, 0, sizeof d);
    int len = decode(bytes, &d);
    if (len != want_len || !d.is_bmi || d.op != want_op ||
        d.opsize != want_opsize || d.dst != want_dst || d.a_src != want_src) {
        printf("FAIL %-34s decode: len=%d op=%d opsize=%d dst=%d src=%d "
               "(want len=%d op=%d opsize=%d dst=%d src=%d)\n",
               name, len, d.op, d.opsize, d.dst, d.a_src,
               want_len, want_op, want_opsize, want_dst, want_src);
        fails++; return;
    }
    uint64_t dst = 0, dst2 = 0, flags = 0;
    if (!bmi_exec(d.op, d.opsize, src_val, 0, &dst, &dst2, &flags)) {
        printf("FAIL %-34s bmi_exec refused\n", name); fails++; return;
    }
    int cf = !!(flags & FLAG_CF), zf = !!(flags & FLAG_ZF);
    if (dst != want_result || cf != want_cf || zf != want_zf) {
        printf("FAIL %-34s src=%#llx -> dst=%llu cf=%d zf=%d "
               "(want dst=%llu cf=%d zf=%d)\n",
               name, (unsigned long long)src_val, (unsigned long long)dst,
               cf, zf, (unsigned long long)want_result, want_cf, want_zf);
        fails++; return;
    }
    printf("PASS %-34s src=%#llx -> %llu cf=%d zf=%d\n",
           name, (unsigned long long)src_val, (unsigned long long)dst, cf, zf);
}

int main(void) {
    /* THE spin instruction: lzcnt cx,di = 66 f3 0f bd cf (5 bytes).
     * dst=rcx(1), src=rdi(7). */
    const uint8_t lz16[] = {0x66, 0xf3, 0x0f, 0xbd, 0xcf};
    /* Inputs chosen to expose the +16 skew of the old 32-bit mis-decode. */
    chk("lzcnt cx,di  src=0x0001", lz16, 5, BMI_LZCNT, 16, 5, 1, 7,
        0x0001, 15, 0, 0);
    chk("lzcnt cx,di  src=0x0080", lz16, 5, BMI_LZCNT, 16, 5, 1, 7,
        0x0080, 8, 0, 0);
    chk("lzcnt cx,di  src=0x8000 (msb)", lz16, 5, BMI_LZCNT, 16, 5, 1, 7,
        0x8000, 0, 0, 1);
    chk("lzcnt cx,di  src=0 (CF)", lz16, 5, BMI_LZCNT, 16, 5, 1, 7,
        0x0000, 16, 1, 0);
    chk("lzcnt cx,di  src=0xFFFF", lz16, 5, BMI_LZCNT, 16, 5, 1, 7,
        0xFFFF, 0, 0, 1);
    /* upper-half garbage in the 64-bit source register must be masked off */
    chk("lzcnt cx,di  hi-garbage masked", lz16, 5, BMI_LZCNT, 16, 5, 1, 7,
        0xDEAD0001, 15, 0, 0);

    /* tzcnt cx,di = 66 f3 0f bc cf */
    const uint8_t tz16[] = {0x66, 0xf3, 0x0f, 0xbc, 0xcf};
    chk("tzcnt cx,di  src=0x8000", tz16, 5, BMI_TZCNT, 16, 5, 1, 7,
        0x8000, 15, 0, 0);
    chk("tzcnt cx,di  src=0x0002", tz16, 5, BMI_TZCNT, 16, 5, 1, 7,
        0x0002, 1, 0, 0);
    chk("tzcnt cx,di  src=0 (CF)", tz16, 5, BMI_TZCNT, 16, 5, 1, 7,
        0x0000, 16, 1, 0);

    /* 32/64-bit forms must be unchanged by the fix */
    const uint8_t lz32[] = {0xf3, 0x0f, 0xbd, 0xcf};          /* lzcnt ecx,edi */
    chk("lzcnt ecx,edi src=1 (32-bit)", lz32, 4, BMI_LZCNT, 32, 4, 1, 7,
        0x1, 31, 0, 0);
    const uint8_t lz64[] = {0xf3, 0x48, 0x0f, 0xbd, 0xcf};    /* lzcnt rcx,rdi */
    chk("lzcnt rcx,rdi src=1 (64-bit)", lz64, 5, BMI_LZCNT, 64, 5, 1, 7,
        0x1, 63, 0, 0);
    const uint8_t tz32[] = {0xf3, 0x0f, 0xbc, 0xcf};          /* tzcnt ecx,edi */
    chk("tzcnt ecx,edi src=4 (32-bit)", tz32, 4, BMI_TZCNT, 32, 4, 1, 7,
        0x4, 2, 0, 0);

    if (fails) { printf("zcnt16: %d FAILURES\n", fails); return 1; }
    printf("zcnt16: all cases pass\n");
    return 0;
}
