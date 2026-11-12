/* M0 interface freeze: hot-path types shared by core / forge / play.
 * Pure C11, no external dependencies. Keep this file stable once frozen.
 */
#ifndef NEURO_TYPES_H
#define NEURO_TYPES_H

#include <stdint.h>
#include <stddef.h>

/* ---------------------------------------------------------------- errors */
enum {
    NEURO_OK                =  0,
    NEURO_E_BAD_ARG         = -1,
    NEURO_E_BAD_MAGIC       = -2,
    NEURO_E_BAD_BOM         = -3,
    NEURO_E_BAD_FLOAT       = -4,
    NEURO_E_BAD_HEADER      = -5,
    NEURO_E_RUNNER_TOO_OLD  = -6,
    NEURO_E_FORMAT_TOO_NEW  = -7,
    NEURO_E_ID_PARSE        = -8,
    NEURO_E_ID_RANGE        = -9
};

/* --------------------------------------------------------------- id parts
 * Readable id: TYPE-BATCH-PARENT-INDIV
 *   TYPE   2..4 ASCII chars
 *   BATCH  4 base36 chars (0-9A-Z)
 *   PARENT 4 base36 chars
 *   INDIV  6 base62 chars (0-9A-Za-z) encoding a 32-bit random code
 * Total length is 19/20/21 for TYPE length 2/3/4.
 * The "18 chars" in the design draft is a one-character miscount.
 */
typedef struct {
    char     type[5];   /* NUL terminated */
    uint32_t batch;
    uint32_t parent;
    uint32_t indiv;     /* 32-bit random code */
} NeuroIdParts;

/* 4 ASCII chars, right padded with 0. "PDc" -> 'P','D','c',0 */
static inline uint32_t neuro_type_code(const char *type) {
    uint32_t c0 = (uint32_t)(uint8_t)type[0];
    uint32_t c1 = (uint32_t)(uint8_t)type[1];
    uint32_t c2 = (uint32_t)(uint8_t)type[2];
    uint32_t c3 = (uint32_t)(uint8_t)type[3];   /* 0 for 3-char types */
    return (c0 << 24) | (c1 << 16) | (c2 << 8) | c3;
}

/* FNV-1a 64 */
static inline uint64_t neuro_fnv1a64(const char *s) {
    uint64_t h = 0xcbf29ce484222325ULL;
    while (*s) {
        h ^= (uint64_t)(uint8_t)*s++;
        h *= 0x100000001b3ULL;
    }
    return h;
}

/* ------------------------------------------------------- hot-path neuron key
 * 24 B. Never used for string formatting on the hot path.
 */
#pragma pack(push, 1)
typedef struct {
    uint64_t id_hash;
    uint32_t type_code;
    uint32_t batch;
    uint32_t parent;
    uint32_t indiv;
} NeuronKey;
#pragma pack(pop)
_Static_assert(sizeof(NeuronKey) == 24, "NeuronKey must be 24 bytes");

/* ------------------------------------------------------ synapse hot record
 * 32 B, contiguous in the hot block. e[] and p_half live side by side.
 * last_tick enables lazy decay: settle only when the synapse is touched.
 */
enum {
    NEURO_SYN_ER_ELIGIBLE  = 1u << 0,
    NEURO_SYN_INHIBITORY   = 1u << 1,
    NEURO_SYN_FROZEN       = 1u << 2,
    NEURO_SYN_PLASTIC_LOCK = 1u << 3
};

#pragma pack(push, 1)
typedef struct {
    uint32_t pre;
    uint32_t post;
    float    w;
    float    e[3];       /* multi-scale eligibility trace, K = 3 */
    uint16_t p_half;     /* metaplasticity, IEEE 754 binary16 */
    uint16_t flags;
    uint32_t last_tick;  /* lazy timestamp for decay */
} SynapseHot;
#pragma pack(pop)
_Static_assert(sizeof(SynapseHot) == 32, "SynapseHot must be 32 bytes");

/* Error-channel isolation: only er_eligible and not frozen synapses respond. */
static inline int neuro_er_allowed(uint16_t flags) {
    return ((flags & NEURO_SYN_ER_ELIGIBLE) != 0) &&
           ((flags & NEURO_SYN_FROZEN) == 0);
}

static inline float neuro_apply_er(float w, float dw, uint16_t flags) {
    return neuro_er_allowed(flags) ? (w + dw) : w;
}

/* ------------------------------------------------------------ PDc class row
 * Posterior is read from relative firing rate, not from a rate layer.
 */
typedef struct {
    uint32_t class_id;
    uint32_t neuron_start;
    uint32_t neuron_count;
    float    prior;
} PdcClass;
_Static_assert(sizeof(PdcClass) == 16, "PdcClass must be 16 bytes");

/* ------------------------------------------------------- id codec (id.c) */
int neuro_id_parse(const char *s, NeuroIdParts *out);
int neuro_id_format(const NeuroIdParts *p, char *buf, size_t cap);
int neuro_neuron_key_make(const NeuroIdParts *p, NeuronKey *out);

#endif /* NEURO_TYPES_H */
