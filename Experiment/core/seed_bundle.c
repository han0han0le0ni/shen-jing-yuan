#include "seed_bundle.h"

uint64_t neuro_splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

void neuro_seed_bundle_init(NeuroSeedBundle *b, uint64_t master) {
    uint64_t s = master;
    b->structure   = neuro_splitmix64(&s);
    b->weight      = neuro_splitmix64(&s);
    b->noise       = neuro_splitmix64(&s);
    b->trial_order = neuro_splitmix64(&s);

    /* SplitMix64 accepts zero states, but keep streams away from degenerate
     * zero to make accidental "no seed" obvious in logs. */
    if (b->structure   == 0) b->structure   = 0x9e3779b97f4a7c15ULL;
    if (b->weight      == 0) b->weight      = 0xbf58476d1ce4e5b9ULL;
    if (b->noise       == 0) b->noise       = 0x94d049bb133111ebULL;
    if (b->trial_order == 0) b->trial_order = 0x2545f4914f6cdd1dULL;
}

uint64_t *neuro_seed_stream_ptr(NeuroSeedBundle *b, NeuroStream s) {
    switch (s) {
    case NEURO_STREAM_STRUCTURE:   return &b->structure;
    case NEURO_STREAM_WEIGHT:      return &b->weight;
    case NEURO_STREAM_NOISE:       return &b->noise;
    case NEURO_STREAM_TRIAL_ORDER: return &b->trial_order;
    default:                       return 0;
    }
}

double neuro_rng_uniform01(uint64_t *state) {
    return (double)(neuro_splitmix64(state) >> 11) * (1.0 / 9007199254740992.0);
}
