/*
 * macho_grow.h — make room in a Mach-O header so the load commands can expand.
 *
 * The problem: tools like change_dylib (and patch_macho's LC_DYLD_INFO_ONLY
 * insertion) write load commands in place, bounded by the file offset of the
 * first section's data. When the linker leaves little padding there (recent
 * Bun/JSC builds leave as few as 16 bytes), a longer dylib path or an extra
 * load command no longer fits.
 *
 * The fix, studied from LIEF (src/MachO/Binary.cpp `shift`) and llvm-objcopy
 * (MachOLayoutBuilder): make room by inserting page-aligned space after the
 * load commands. LIEF/llvm push every later segment to a HIGHER vm address and
 * then fix up everything that depended on those addresses — section-symbol
 * n_values, LC_MAIN, function-start deltas, relocations, rebase/bind/chained
 * targets. That is a lot of machinery.
 *
 * We take a simpler, equivalent route available to any PIE executable with a
 * __PAGEZERO: instead of raising data, we LOWER the image base. We donate the
 * inserted bytes from __PAGEZERO and drop __TEXT's vmaddr by the same amount,
 * growing __TEXT's vm/file size. Net effect: every section and segment keeps
 * its ORIGINAL vm address, so no pointer, rebase, bind, n_value, or entry
 * address ever changes. The only fields that move are file offsets — which we
 * shift uniformly. (Borrowed from LIEF: the exhaustive list of offset fields.)
 *
 * Precondition: a MH_PIE executable with a __PAGEZERO at least `grow` bytes
 * large. (Always true for the Claude Code executable: 0x1_0000_0000 pagezero.)
 * Dylibs without a __PAGEZERO can't lower the base; mg_grow_header reports that
 * and leaves the buffer untouched so the caller can fall back / error cleanly.
 */
#ifndef MACHO_GROW_H
#define MACHO_GROW_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <mach-o/loader.h>

/* Load-command constants newer than the 10.9 SDK headers. */
#ifndef LC_DYLD_EXPORTS_TRIE
#define LC_DYLD_EXPORTS_TRIE        0x80000033
#endif
#ifndef LC_DYLD_CHAINED_FIXUPS
#define LC_DYLD_CHAINED_FIXUPS      0x80000034
#endif
#ifndef LC_DYLIB_CODE_SIGN_DRS
#define LC_DYLIB_CODE_SIGN_DRS      0x2B
#endif
#ifndef LC_LINKER_OPTIMIZATION_HINT
#define LC_LINKER_OPTIMIZATION_HINT 0x2E
#endif

#define MG_PAGE 0x1000UL

/* Lowest section file offset — this bounds the header pad. */
static uint32_t mg_first_sect_off(const uint8_t *buf) {
    const struct mach_header_64 *hdr = (const struct mach_header_64 *)buf;
    uint32_t first = UINT32_MAX;
    const uint8_t *lcp = buf + sizeof(*hdr);
    for (uint32_t i = 0; i < hdr->ncmds; i++) {
        const struct load_command *lc = (const struct load_command *)lcp;
        if (lc->cmd == LC_SEGMENT_64) {
            const struct segment_command_64 *seg = (const struct segment_command_64 *)lcp;
            const struct section_64 *sect = (const struct section_64 *)(lcp + sizeof(*seg));
            for (uint32_t j = 0; j < seg->nsects; j++)
                if (sect[j].offset && sect[j].offset < first) first = sect[j].offset;
        }
        lcp += lc->cmdsize;
    }
    return first == UINT32_MAX ? 4096 : first;
}

/* Shift one file-offset field down by `grow` if it points at/after `insert`. */
static void mg_bump(uint32_t *off, uint32_t insert, uint32_t grow) {
    if (*off >= insert) *off += grow;
}

/*
 * Grow the header pad by at least `grow_req` bytes (rounded up to a page).
 * pbuf is realloc'd, pfsize updated. Returns 0 on success, -1 if the
 * precondition (PIE-style __PAGEZERO large enough) isn't met — in which case
 * the buffer and size are left unchanged.
 */
static int mg_grow_header(uint8_t **pbuf, size_t *pfsize, uint32_t grow_req) {
    uint8_t *buf = *pbuf;
    size_t fsize = *pfsize;
    struct mach_header_64 *hdr = (struct mach_header_64 *)buf;

    uint32_t grow = (uint32_t)((grow_req + MG_PAGE - 1) & ~(MG_PAGE - 1));
    if (grow == 0) return 0;

    /* Validate preconditions before mutating anything. The image-base trick is
     * only sound for a PIE executable: it needs a __PAGEZERO to donate vm space
     * from, and it relies on the image being position-independent so that
     * lowering the base (every address shifts by the same amount) is a no-op at
     * load time. A non-PIE image, or a dylib/bundle (no __PAGEZERO), would
     * require fixing up absolute pointers — which this tool deliberately does
     * not do. Refuse loudly rather than silently corrupt. */
    if (hdr->magic != MH_MAGIC_64) {
        fprintf(stderr, "macho_grow: not a 64-bit Mach-O (magic=0x%x)\n", hdr->magic);
        return -1;
    }
    if (hdr->filetype != MH_EXECUTE) {
        fprintf(stderr, "macho_grow: only MH_EXECUTE is supported (filetype=%u); the "
                        "image-base trick needs a __PAGEZERO. Use the heavyweight "
                        "shift-up approach for dylibs/bundles (see HEADER_PAD_GROWTH.md)\n",
                hdr->filetype);
        return -1;
    }
    if (!(hdr->flags & MH_PIE)) {
        fprintf(stderr, "macho_grow: executable is not PIE (flags=0x%x); lowering the "
                        "image base would require fixing absolute relocations, which "
                        "this tool does not do (see HEADER_PAD_GROWTH.md)\n", hdr->flags);
        return -1;
    }

    uint32_t insert = mg_first_sect_off(buf);

    /* We insert space at `insert` (the first section's file offset) and shift
     * everything from there onward. That point must be at/after the end of the
     * load commands, or we'd memmove the tail of the LC table itself. A healthy
     * binary always satisfies this; refuse the anomalous case rather than
     * corrupt it. */
    uint32_t lc_end = (uint32_t)sizeof(*hdr) + hdr->sizeofcmds;
    if (insert < lc_end) {
        fprintf(stderr, "macho_grow: first section (%u) precedes end of load commands "
                        "(%u); refusing to grow a malformed header\n", insert, lc_end);
        return -1;
    }

    /* Locate the donor (__PAGEZERO) and the header-bearing segment (__TEXT). */
    struct segment_command_64 *pagezero = NULL, *text = NULL;
    uint8_t *lcp = buf + sizeof(*hdr);
    for (uint32_t i = 0; i < hdr->ncmds; i++) {
        struct load_command *lc = (struct load_command *)lcp;
        if (lc->cmd == LC_SEGMENT_64) {
            struct segment_command_64 *seg = (struct segment_command_64 *)lcp;
            if (strcmp(seg->segname, "__PAGEZERO") == 0) pagezero = seg;
            else if (seg->fileoff == 0 && seg->filesize > 0) text = seg;
        }
        lcp += lc->cmdsize;
    }
    if (!text) {
        fprintf(stderr, "macho_grow: no __TEXT-like segment holds the header\n");
        return -1;
    }
    if (!pagezero || pagezero->vmsize < grow) {
        fprintf(stderr, "macho_grow: need a __PAGEZERO >= %u bytes to lower the image "
                        "base (image-base trick requires a PIE executable)\n", grow);
        return -1;
    }

    /* Insert `grow` zero bytes after the load commands, shifting file data down. */
    uint8_t *nbuf = (uint8_t *)realloc(buf, fsize + grow);
    if (!nbuf) { fprintf(stderr, "macho_grow: realloc failed\n"); return -1; }
    buf = nbuf;
    hdr = (struct mach_header_64 *)buf;
    memmove(buf + insert + grow, buf + insert, fsize - insert);
    memset(buf + insert, 0, grow);

    /* Patch the header. Load commands live before `insert`, so memmove didn't
     * touch them; we walk them now and adjust only file-offset fields, plus the
     * three VM fields that keep every address fixed. */
    lcp = buf + sizeof(*hdr);
    for (uint32_t i = 0; i < hdr->ncmds; i++) {
        struct load_command *lc = (struct load_command *)lcp;
        switch (lc->cmd) {
        case LC_SEGMENT_64: {
            struct segment_command_64 *seg = (struct segment_command_64 *)lcp;
            /* Identify segments by criteria, not by a saved pointer: realloc
             * above may have moved the buffer, invalidating the pointers found
             * during validation. The header-bearing segment is the one mapped
             * at file offset 0 with content (i.e. __TEXT, not __PAGEZERO). */
            if (strcmp(seg->segname, "__PAGEZERO") == 0) {
                seg->vmsize -= grow;            /* donate space below __TEXT */
            } else if (seg->fileoff == 0 && seg->filesize > 0) {
                seg->vmaddr  -= grow;           /* lower the image base */
                seg->vmsize  += grow;
                seg->filesize += grow;          /* fileoff stays 0 */
            } else if (seg->fileoff >= insert) {
                seg->fileoff += grow;           /* later segment: file moves, vm fixed */
            }
            struct section_64 *sect = (struct section_64 *)(lcp + sizeof(*seg));
            for (uint32_t j = 0; j < seg->nsects; j++) {
                mg_bump(&sect[j].offset, insert, grow);   /* addr stays fixed */
                if (sect[j].reloff) mg_bump(&sect[j].reloff, insert, grow);
            }
            break;
        }
        case LC_SYMTAB: {
            struct symtab_command *c = (struct symtab_command *)lcp;
            mg_bump(&c->symoff, insert, grow);
            mg_bump(&c->stroff, insert, grow);
            break;
        }
        case LC_DYSYMTAB: {
            struct dysymtab_command *c = (struct dysymtab_command *)lcp;
            mg_bump(&c->tocoff, insert, grow);
            mg_bump(&c->modtaboff, insert, grow);
            mg_bump(&c->extrefsymoff, insert, grow);
            mg_bump(&c->indirectsymoff, insert, grow);
            mg_bump(&c->extreloff, insert, grow);
            mg_bump(&c->locreloff, insert, grow);
            break;
        }
        case LC_DYLD_INFO:
        case LC_DYLD_INFO_ONLY: {
            struct dyld_info_command *c = (struct dyld_info_command *)lcp;
            mg_bump(&c->rebase_off, insert, grow);
            mg_bump(&c->bind_off, insert, grow);
            mg_bump(&c->weak_bind_off, insert, grow);
            mg_bump(&c->lazy_bind_off, insert, grow);
            mg_bump(&c->export_off, insert, grow);
            break;
        }
        case LC_MAIN: {
            /* entryoff is a file offset within __TEXT; bumping it keeps the
             * entry's vm address fixed (base went down by the same amount). */
            struct entry_point_command *c = (struct entry_point_command *)lcp;
            uint32_t e = (uint32_t)c->entryoff;
            mg_bump(&e, insert, grow);
            c->entryoff = e;
            break;
        }
        case LC_FUNCTION_STARTS:
        case LC_DATA_IN_CODE:
        case LC_CODE_SIGNATURE:
        case LC_SEGMENT_SPLIT_INFO:
        case LC_DYLIB_CODE_SIGN_DRS:
        case LC_LINKER_OPTIMIZATION_HINT:
        case LC_DYLD_EXPORTS_TRIE:
        case LC_DYLD_CHAINED_FIXUPS: {
            struct linkedit_data_command *c = (struct linkedit_data_command *)lcp;
            mg_bump(&c->dataoff, insert, grow);
            break;
        }
        default:
            break;  /* LC_LOAD_DYLIB/DYLINKER/UUID/VERSION_MIN carry no file offsets */
        }
        lcp += lc->cmdsize;
    }

    *pbuf = buf;
    *pfsize = fsize + grow;
    return 0;
}

#endif /* MACHO_GROW_H */
