/* .neuro container header and block table (M0 freeze).
 * Rules:
 *   1. Endianness / float format are validated BEFORE any pointer cast.
 *   2. Hot blocks stay uncompressed and are mmap-ready.
 *   3. Cold blocks may be compressed; sizes are recorded per block.
 *   4. A newer format is never parsed best-effort.
 */
#ifndef NEURO_CONTAINER_NEURO_H
#define NEURO_CONTAINER_NEURO_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "neuro_types.h"
#include "seed_bundle.h"

#define NEURO_MAGIC_SIZE          8
#define NEURO_FORMAT_VERSION      1u
#define NEURO_MIN_RUNNER_VERSION  1u
#define NEURO_BOM_LE              0x01u
#define NEURO_FLOAT_IEEE754       0x01u

#define NEURO_HEADER_SIZE         72u
#define NEURO_BLOCK_ENTRY_SIZE    48u

typedef enum {
    NEURO_BLOCK_STRING_POOL      = 1,   /* cold  */
    NEURO_BLOCK_METADATA         = 2,   /* cold  */
    NEURO_BLOCK_WEIGHT           = 3,   /* hot   */
    NEURO_BLOCK_ELIGIBILITY_TRACE= 4,   /* hot   */
    NEURO_BLOCK_METAPLASTICITY   = 5,   /* hot   */
    NEURO_BLOCK_MEMORY_SLOTS     = 6,   /* cold  */
    NEURO_BLOCK_DECODER_STATE    = 7,   /* cold  */
    NEURO_BLOCK_SNC_RECORDS      = 8,   /* cold  */
    NEURO_BLOCK_PRIMITIVE_LIBRARY= 9,   /* cold  */
    NEURO_BLOCK_COUNT            = 9
} NeuroBlockType;

enum {
    NEURO_BLOCK_FLAG_HOT        = 1u << 0,
    NEURO_BLOCK_FLAG_COMPRESSED = 1u << 1,
    NEURO_BLOCK_FLAG_ZERO_COPY  = 1u << 2
};

#pragma pack(push, 1)
typedef struct {
    uint8_t  magic[NEURO_MAGIC_SIZE];  /* "NEURO\0\0\1" */
    uint8_t  bom;                      /* NEURO_BOM_LE */
    uint8_t  float_format;             /* NEURO_FLOAT_IEEE754 */
    uint16_t format_version;
    uint16_t min_runner_version;
    uint16_t header_size;              /* sizeof(NeuroHeader) */
    NeuroSeedBundle seed;              /* 4 x u64, 32 B */
    uint8_t  config_digest[8];
    uint64_t index_offset;
    uint32_t block_count;
    uint32_t reserved;
} NeuroHeader;
#pragma pack(pop)
_Static_assert(sizeof(NeuroHeader) == NEURO_HEADER_SIZE, "NeuroHeader size");

#pragma pack(push, 1)
typedef struct {
    uint32_t type;        /* NeuroBlockType */
    uint32_t flags;       /* NEURO_BLOCK_FLAG_* */
    uint64_t offset;      /* absolute file offset */
    uint64_t comp_size;   /* bytes on disk */
    uint64_t raw_size;    /* bytes after decompression */
    uint64_t count;       /* element count, 0 for opaque blobs */
    uint32_t elem_size;   /* bytes per element, 0 for opaque blobs */
    uint32_t crc32;       /* CRC32 of comp bytes */
} NeuroBlockEntry;
#pragma pack(pop)
_Static_assert(sizeof(NeuroBlockEntry) == NEURO_BLOCK_ENTRY_SIZE, "block entry size");

static inline int neuro_host_is_little_endian(void) {
    uint16_t x = 1;
    return (*(const uint8_t *)&x) == 1;
}

/* Cheap probe: safe to call on the first 10 bytes before mmap casting. */
static inline int neuro_magic_probe(const uint8_t *raw) {
    static const uint8_t k_magic[NEURO_MAGIC_SIZE] =
        { 'N', 'E', 'U', 'R', 'O', 0x00, 0x00, 0x01 };
    if (!raw) return NEURO_E_BAD_ARG;
    if (memcmp(raw, k_magic, NEURO_MAGIC_SIZE) != 0) return NEURO_E_BAD_MAGIC;
    if (raw[8] != NEURO_BOM_LE) return NEURO_E_BAD_BOM;
    if (raw[9] != NEURO_FLOAT_IEEE754) return NEURO_E_BAD_FLOAT;
    return NEURO_OK;
}

static inline int neuro_header_validate(const NeuroHeader *h) {
    static const uint8_t k_magic[NEURO_MAGIC_SIZE] =
        { 'N', 'E', 'U', 'R', 'O', 0x00, 0x00, 0x01 };
    if (!h) return NEURO_E_BAD_ARG;
    if (memcmp(h->magic, k_magic, NEURO_MAGIC_SIZE) != 0) return NEURO_E_BAD_MAGIC;
    if (h->bom != NEURO_BOM_LE) return NEURO_E_BAD_BOM;
    if (h->float_format != NEURO_FLOAT_IEEE754) return NEURO_E_BAD_FLOAT;
    if (h->header_size != (uint16_t)sizeof(NeuroHeader)) return NEURO_E_BAD_HEADER;
    if (h->block_count >= NEURO_BLOCK_COUNT) return NEURO_E_BAD_HEADER;
    if (h->index_offset < (uint64_t)sizeof(NeuroHeader)) return NEURO_E_BAD_HEADER;
    return NEURO_OK;
}

/* Version gate. Both checks are intentional: no downgrade, no best effort. */
static inline int neuro_header_check_runner(const NeuroHeader *h,
                                            uint16_t runner_version) {
    if (!h) return NEURO_E_BAD_ARG;
    if (runner_version < h->min_runner_version) return NEURO_E_RUNNER_TOO_OLD;
    if (h->format_version > NEURO_FORMAT_VERSION) return NEURO_E_FORMAT_TOO_NEW;
    return NEURO_OK;
}

/* Hot block byte size for a given element count. */
static inline uint64_t neuro_synapse_block_bytes(uint64_t synapse_count) {
    return synapse_count * (uint64_t)sizeof(SynapseHot);
}

/* Checkpoint rollback note: eligibility traces are NOT rolled back.
 * Callers must therefore settle (decay) traces by the wall/tick delta after a
 * rollback, or clear them, before the next weight update. */
static inline void neuro_trace_after_rollback(float e[3], float decay_per_tick,
                                              uint32_t delta_ticks) {
    if (!e) return;
    float f = 1.0f;
    for (uint32_t i = 0; i < delta_ticks; i++) {
        f *= decay_per_tick;
        if (f < 1e-6f) { f = 0.0f; break; }
    }
    for (int k = 0; k < 3; k++) e[k] *= f;
}

#endif /* NEURO_CONTAINER_NEURO_H */
