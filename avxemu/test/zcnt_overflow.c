/*
 * zcnt_overflow.c — regression test for the lde_rd_zcnt worklist overflow.
 *
 * lde_rd_zcnt walks a function's CFG with an n-slot stack (n = fend - fstart).
 * The bug: "visited" was marked on POP, and the pushes had no `sp < n` guard, so
 * a node reachable many times before it is popped (high fan-in, or a dense
 * switch table with repeated entries) was pushed once per in-edge and overran
 * the stack — a heap-buffer-overflow write that corrupts adjacent malloc
 * metadata (seen in the wild as Node/Bun crashing with
 * "incorrect checksum for freed object" at dylib load).
 *
 * This builds a 10-byte "function":
 *      lea  9(%rip), %rax     ; table base = offset 16
 *      jmp  *%rax             ; indirect jump (switch dispatch)
 *      ret                    ; offset 9 — a valid in-range target
 * followed by a 20-entry jump table at offset 16 whose every entry points at
 * the ret (offset 9). The scanner resolves the table and, pre-fix, pushes
 * offset 9 twenty times into a 10-slot stack -> overflow.
 *
 * Run under Guard Malloc so the overrun faults deterministically:
 *   DYLD_INSERT_LIBRARIES=/usr/lib/libgmalloc.dylib ./zcnt_overflow
 * Pre-fix: SIGSEGV.  Post-fix: prints OK and exits 0.
 */
#include "lde.h"
#include <stdint.h>
#include <string.h>
#include <stdio.h>

int main(void) {
    uint8_t buf[256];
    memset(buf, 0x90, sizeof buf);                 /* nop padding */

    /* lea 9(%rip), %rax  ->  last_lea = (0+7) + 9 = 16 */
    buf[0] = 0x48; buf[1] = 0x8D; buf[2] = 0x05;
    int32_t disp = 9; memcpy(buf + 3, &disp, 4);
    /* jmp *%rax */
    buf[7] = 0xFF; buf[8] = 0xE0;
    /* ret (a valid branch target inside [0,10)) */
    buf[9] = 0xC3;

    /* 20-entry table at offset 16; each entry = (9 - 16) so target = 9 */
    int32_t entry = 9 - 16;
    for (int i = 0; i < 20; i++) memcpy(buf + 16 + 4 * i, &entry, 4);

    size_t out[256];
    long cnt = lde_rd_zcnt(buf, /*fstart*/0, /*fend*/10, /*readable*/sizeof buf, out, 256);

    printf("OK: lde_rd_zcnt returned %ld (no stack overflow)\n", cnt);
    return 0;
}
