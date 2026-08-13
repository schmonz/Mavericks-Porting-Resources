#ifndef AVXEMU_REGFILE_H
#define AVXEMU_REGFILE_H

#include <stdint.h>
#include <stddef.h>
#include "decode.h"

/*
 * A flat snapshot of the machine state the emulator needs. Both entry points
 * fill one and hand it to avxemu_emulate():
 *   - the SIGILL handler copies it out of the signal's AVX mcontext;
 *   - the trampoline saver (tramp.s) spills the live registers into it.
 * ymm is contiguous 32-byte (low xmm + high half), gpr is x86 register order
 * (0=rax,1=rcx,2=rdx,3=rbx,4=rsp,5=rbp,6=rsi,7=rdi,8..15=r8..r15).
 */
typedef struct {
    uint8_t  ymm[16][32];
    uint64_t gpr[16];
    uint64_t rflags;
    uint64_t rip;        /* address of the instruction (EA base for rip-rel; resume = rip + d->len) */
} avxemu_regfile;

/* Emulate one decoded instruction against rf, in place. Returns 1 on success,
 * 0 if the op is recognised-but-unimplemented (caller should fall back). */
int avxemu_emulate(const decoded *d, avxemu_regfile *rf);

/* Block-window relocation: relocate the faulting instruction at `site` (plus
 * enough following position-independent instructions to reach >=5 bytes) into the
 * RWX code cache, then patch `site` with a 5-byte jmp into it. The faulting
 * instruction runs via avxemu_emulate; the following instructions are copied
 * verbatim. Returns 1 if relocated, 0 if it could not be done safely (caller
 * should keep using the SIGILL emulation path -- no regression). */
int avxemu_relocate_block(uint8_t *site);  /* returns 1 if relocated */

/* Raw 16-byte-aligned bump allocation from the SINGLE shared RWX thunk pool
 * (tramp.c's g_pool/g_used). Both the eager trampoliner and the runtime relocator
 * allocate through this one cursor so they can never hand out overlapping bytes.
 * Returns NULL if the pool is uninitialized or exhausted. */
void *avxemu_pool_alloc(size_t n);

/* Window patch-safety (Part 3): is it safe to write a `jmp` of `jmplen` bytes at
 * `site`? Finds the function containing `site` (cached LC_FUNCTION_STARTS bounds)
 * and returns 0 if any branch target lands in the open interval (site, site+jmplen)
 * OR the containing function can't be determined/decoded (conservative decline).
 * Returns 1 only if provably safe. */
int avxemu_patch_safe(uint8_t *site, int jmplen);

#endif /* AVXEMU_REGFILE_H */
