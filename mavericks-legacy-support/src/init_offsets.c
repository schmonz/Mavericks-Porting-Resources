/*
 * Run the main executable's __TEXT,__init_offsets initializers.
 *
 * dyld4-era toolchains emit C/C++ static constructors into a section of type
 * S_INIT_FUNC_OFFSETS ("__TEXT,__init_offsets"), which holds 4-byte offsets
 * from the image base rather than the 8-byte pointers of the older
 * S_MOD_INIT_FUNC_POINTERS ("__DATA,__mod_init_func"). 10.9's dyld only
 * recognizes the pointer form, so it walks past __init_offsets without a
 * word of complaint and the constructors simply never run.
 *
 * The failure that follows is a null dereference somewhere deep in a library
 * that assumed its own initializer had already run: allocator roots, lock
 * words and vtable pointers are all still zero. Bun's mimalloc, for instance,
 * reaches main() with an uninitialized subproc record and faults on the first
 * mi_heap_new().
 *
 * This constructor runs late (after the rest of this library's constructors,
 * so the polyfills the initializers may call are already live), finds the
 * MH_EXECUTE image, and invokes each entry in its __init_offsets.
 *
 * 10.9's libobjc has the matching blind spot for the same class of binaries:
 * it doesn't rewrite their __objc_selrefs entries from c-string pointers into
 * runtime SELs, so objc_msgSend dispatches on a string pointer and the runtime
 * reports a selector mismatch. We register those selectors ourselves in the
 * same pass.
 */
#include <stdint.h>
#include <string.h>
#include <mach-o/loader.h>
#include <mach-o/dyld.h>
#include <objc/objc.h>
#include <objc/runtime.h>

/* Highest priority value, so this runs after every default-priority
 * constructor in this library. */
__attribute__((constructor(65535)))
static void mav_run_init_offsets(void) {
    static int done = 0;
    if (done) return;
    done = 1;

    /* Locate the main executable. Index 0 is conventionally the main image,
     * but a DYLD_INSERT_LIBRARIES load can precede it, so match on filetype. */
    const struct mach_header_64 *hdr = NULL;
    intptr_t slide = 0;
    uint32_t count = _dyld_image_count();
    for (uint32_t i = 0; i < count; i++) {
        const struct mach_header *mh = _dyld_get_image_header(i);
        if (mh && mh->magic == MH_MAGIC_64 && mh->filetype == MH_EXECUTE) {
            hdr = (const struct mach_header_64 *)mh;
            slide = _dyld_get_image_vmaddr_slide(i);
            break;
        }
    }
    if (!hdr) return;

    const uint8_t *lcp = (const uint8_t *)(hdr + 1);
    for (uint32_t i = 0; i < hdr->ncmds; i++) {
        const struct load_command *lc = (const struct load_command *)lcp;
        if (lc->cmd == LC_SEGMENT_64) {
            const struct segment_command_64 *seg =
                (const struct segment_command_64 *)lc;
            const struct section_64 *sect =
                (const struct section_64 *)((const uint8_t *)seg + sizeof(*seg));
            for (uint32_t j = 0; j < seg->nsects; j++) {
                if (strcmp(sect[j].sectname, "__objc_selrefs") == 0) {
                    SEL *selrefs = (SEL *)(uintptr_t)(sect[j].addr + slide);
                    uint64_t n = sect[j].size / sizeof(SEL);
                    for (uint64_t k = 0; k < n; k++) {
                        const char *name = (const char *)selrefs[k];
                        if (name) selrefs[k] = sel_registerName(name);
                    }
                } else if (strcmp(sect[j].sectname, "__init_offsets") == 0) {
                    /* Offsets are measured from the image base, which is where
                     * the mach_header sits. */
                    const uint32_t *offsets =
                        (const uint32_t *)(uintptr_t)(sect[j].addr + slide);
                    uint64_t n = sect[j].size / sizeof(uint32_t);
                    for (uint64_t k = 0; k < n; k++) {
                        void (*fn)(void) =
                            (void (*)(void))((uintptr_t)hdr + offsets[k]);
                        fn();
                    }
                }
            }
        }
        lcp += lc->cmdsize;
    }
}
