/* seed_bundle: four independent random streams derived from one master seed.
 * structure / weight / noise / trial_order must never share a global RNG.
 */
#ifndef NEURO_SEED_BUNDLE_H
#define NEURO_SEED_BUNDLE_H

#include <stdint.h>

typedef struct {
    uint64_t structure;
    uint64_t weight;
    uint64_t noise;
    uint64_t trial_order;
} NeuroSeedBundle;

typedef enum {
    NEURO_STREAM_STRUCTURE = 0,
    NEURO_STREAM_WEIGHT,
    NEURO_STREAM_NOISE,
    NEURO_STREAM_TRIAL_ORDER,
    NEURO_STREAM_COUNT
} NeuroStream;

/* SplitMix64 */
uint64_t neuro_splitmix64(uint64_t *state);

/* Derive the four streams; deterministic for a given master seed. */
void neuro_seed_bundle_init(NeuroSeedBundle *b, uint64_t master);

/* Pointer to one stream, for callers that need a specific stream. */
uint64_t *neuro_seed_stream_ptr(NeuroSeedBundle *b, NeuroStream s);

/* Uniform in [0,1) with 53-bit resolution. */
double neuro_rng_uniform01(uint64_t *state);

#endif /* NEURO_SEED_BUNDLE_H */
