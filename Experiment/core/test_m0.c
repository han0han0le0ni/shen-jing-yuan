/* M0 acceptance test: seed bundle, id codec, hot layout, container header,
 * er_eligible isolation, PDc row size. No external dependency beyond libm.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stddef.h>

#include "neuro_types.h"
#include "seed_bundle.h"
#include "container_neuro.h"

static int g_fail = 0;
static int g_checks = 0;

#define CHECK(cond) do { \
    g_checks++; \
    if (!(cond)) { \
        printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
        g_fail++; \
    } \
} while (0)

static void test_seed(void) {
    NeuroSeedBundle a, b, c;
    neuro_seed_bundle_init(&a, 12345u);
    neuro_seed_bundle_init(&b, 12345u);
    neuro_seed_bundle_init(&c, 12346u);

    CHECK(a.structure == b.structure);
    CHECK(a.weight == b.weight);
    CHECK(a.noise == b.noise);
    CHECK(a.trial_order == b.trial_order);
    CHECK(c.structure != a.structure);

    /* the four streams must be distinct */
    CHECK(a.structure != a.weight);
    CHECK(a.weight != a.noise);
    CHECK(a.noise != a.trial_order);

    /* streams advance independently */
    NeuroSeedBundle x = a, y = a;
    for (int i = 0; i < 1000; i++) (void)neuro_splitmix64(&x.noise);
    CHECK(x.noise != y.noise);
    CHECK(x.structure == y.structure);

    /* uniform sanity: mean 0.5, var 1/12 */
    NeuroSeedBundle u;
    neuro_seed_bundle_init(&u, 7u);
    double sum = 0.0, sum2 = 0.0;
    const int N = 20000;
    for (int i = 0; i < N; i++) {
        double v = neuro_rng_uniform01(&u.weight);
        sum += v;
        sum2 += v * v;
    }
    double mean = sum / N;
    double var = sum2 / N - mean * mean;
    CHECK(fabs(mean - 0.5) < 0.01);
    CHECK(fabs(var - (1.0 / 12.0)) < 0.005);

    /* selector */
    CHECK(neuro_seed_stream_ptr(&u, NEURO_STREAM_STRUCTURE) == &u.structure);
    CHECK(neuro_seed_stream_ptr(&u, NEURO_STREAM_TRIAL_ORDER) == &u.trial_order);
    CHECK(neuro_seed_stream_ptr(&u, NEURO_STREAM_COUNT) == NULL);
}

static void test_id(void) {
    NeuroIdParts p;
    char buf[32];
    NeuroIdParts q;
    NeuronKey key;

    memset(&p, 0, sizeof(p));
    strcpy(p.type, "SVI");
    p.batch = 12345u;
    p.parent = 678u;
    p.indiv = 0xDEADBEEFu;

    CHECK(neuro_id_format(&p, buf, sizeof(buf)) == NEURO_OK);
    CHECK(strlen(buf) == 20);   /* 3 + 1 + 4 + 1 + 4 + 1 + 6 */

    memset(&q, 0, sizeof(q));
    CHECK(neuro_id_parse(buf, &q) == NEURO_OK);
    CHECK(strcmp(q.type, "SVI") == 0);
    CHECK(q.batch == p.batch);
    CHECK(q.parent == p.parent);
    CHECK(q.indiv == p.indiv);

    CHECK(neuro_neuron_key_make(&p, &key) == NEURO_OK);
    CHECK(key.type_code == neuro_type_code("SVI"));
    CHECK(key.id_hash == neuro_fnv1a64(buf));
    CHECK(key.batch == p.batch && key.parent == p.parent && key.indiv == p.indiv);

    /* 2-char type -> 19 chars, 4-char type -> 21 chars */
    p.type[0] = 'G'; p.type[1] = 'R'; p.type[2] = '\0';
    CHECK(neuro_id_format(&p, buf, sizeof(buf)) == NEURO_OK);
    CHECK(strlen(buf) == 19);
    CHECK(neuro_id_parse(buf, &q) == NEURO_OK && strcmp(q.type, "GR") == 0);

    strcpy(p.type, "SVIL");
    CHECK(neuro_id_format(&p, buf, sizeof(buf)) == NEURO_OK);
    CHECK(strlen(buf) == 21);

    /* hash properties: stable and case sensitive */
    CHECK(neuro_fnv1a64("SVI-0001-0002-000003") ==
          neuro_fnv1a64("SVI-0001-0002-000003"));
    CHECK(neuro_fnv1a64("SVI") != neuro_fnv1a64("svi"));

    /* reject malformed ids */
    CHECK(neuro_id_parse("SVI-0001-0002", &q) != NEURO_OK);
    CHECK(neuro_id_parse("S-0001-0002-000003", &q) != NEURO_OK);
    CHECK(neuro_id_parse("SVILX-0001-0002-000003", &q) != NEURO_OK);
    CHECK(neuro_id_parse("SVI-00!1-0002-000003", &q) != NEURO_OK);
    CHECK(neuro_id_parse("SVI-0001-0002-00000", &q) != NEURO_OK);
    CHECK(neuro_id_parse("SVI-0001-0002-000003-EXTRA", &q) != NEURO_OK);

    /* capacity guard */
    char tiny[8];
    CHECK(neuro_id_format(&p, tiny, sizeof(tiny)) == NEURO_E_BAD_ARG);
}

static void test_layout(void) {
    CHECK(sizeof(NeuronKey) == 24);
    CHECK(sizeof(SynapseHot) == 32);
    CHECK(sizeof(PdcClass) == 16);
    CHECK(sizeof(NeuroHeader) == NEURO_HEADER_SIZE);
    CHECK(sizeof(NeuroBlockEntry) == NEURO_BLOCK_ENTRY_SIZE);

    CHECK(offsetof(SynapseHot, w) == 8);
    CHECK(offsetof(SynapseHot, e) == 12);
    CHECK(offsetof(SynapseHot, p_half) == 24);
    CHECK(offsetof(SynapseHot, flags) == 26);
    CHECK(offsetof(SynapseHot, last_tick) == 28);

    CHECK(neuro_synapse_block_bytes(1000000ull) == 32000000ull);
    CHECK(neuro_host_is_little_endian() == 1);
}

static void test_container(void) {
    static const uint8_t k_magic[NEURO_MAGIC_SIZE] =
        { 'N', 'E', 'U', 'R', 'O', 0x00, 0x00, 0x01 };
    NeuroHeader h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, k_magic, NEURO_MAGIC_SIZE);
    h.bom = NEURO_BOM_LE;
    h.float_format = NEURO_FLOAT_IEEE754;
    h.format_version = NEURO_FORMAT_VERSION;
    h.min_runner_version = NEURO_MIN_RUNNER_VERSION;
    h.header_size = (uint16_t)sizeof(NeuroHeader);
    h.block_count = 3;
    h.index_offset = sizeof(NeuroHeader);
    neuro_seed_bundle_init(&h.seed, 42u);

    CHECK(neuro_header_validate(&h) == NEURO_OK);
    CHECK(neuro_header_check_runner(&h, 1) == NEURO_OK);
    CHECK(neuro_header_check_runner(&h, 0) == NEURO_E_RUNNER_TOO_OLD);

    h.format_version = 99;
    CHECK(neuro_header_check_runner(&h, 1) == NEURO_E_FORMAT_TOO_NEW);
    h.format_version = NEURO_FORMAT_VERSION;

    /* probe only needs the first 10 bytes */
    uint8_t raw[10];
    memcpy(raw, k_magic, 8);
    raw[8] = NEURO_BOM_LE;
    raw[9] = NEURO_FLOAT_IEEE754;
    CHECK(neuro_magic_probe(raw) == NEURO_OK);
    raw[9] = 0x02;
    CHECK(neuro_magic_probe(raw) == NEURO_E_BAD_FLOAT);
    raw[8] = 0x02;
    CHECK(neuro_magic_probe(raw) == NEURO_E_BAD_BOM);

    h.bom = 0x02;
    CHECK(neuro_header_validate(&h) == NEURO_E_BAD_BOM);
    h.bom = NEURO_BOM_LE;

    h.header_size = 64;
    CHECK(neuro_header_validate(&h) == NEURO_E_BAD_HEADER);
    h.header_size = (uint16_t)sizeof(NeuroHeader);

    h.block_count = NEURO_BLOCK_COUNT;
    CHECK(neuro_header_validate(&h) == NEURO_E_BAD_HEADER);
    h.block_count = 3;

    h.index_offset = 8;
    CHECK(neuro_header_validate(&h) == NEURO_E_BAD_HEADER);
}

static void test_er_isolation(void) {
    /* Only er_eligible and not frozen synapses may receive error updates. */
    CHECK(neuro_apply_er(1.0f, 0.1f, 0) == 1.0f);
    CHECK(fabsf(neuro_apply_er(1.0f, 0.1f, NEURO_SYN_ER_ELIGIBLE) - 1.1f) < 1e-6f);
    CHECK(neuro_apply_er(1.0f, 0.1f, NEURO_SYN_ER_ELIGIBLE | NEURO_SYN_FROZEN) == 1.0f);
    CHECK(neuro_apply_er(1.0f, 0.1f, NEURO_SYN_INHIBITORY) == 1.0f);
    CHECK(neuro_er_allowed(NEURO_SYN_ER_ELIGIBLE) == 1);
    CHECK(neuro_er_allowed(0) == 0);

    /* rollback: traces decay, never silently keep old credit */
    float e[3] = { 1.0f, 1.0f, 1.0f };
    neuro_trace_after_rollback(e, 0.5f, 10);
    CHECK(fabsf(e[0] - 0.0009765625f) < 1e-9f);
    neuro_trace_after_rollback(e, 0.0f, 1);
    CHECK(e[0] == 0.0f && e[1] == 0.0f && e[2] == 0.0f);
}

int main(void) {
    printf("M0 acceptance test\n");
    test_seed();
    test_id();
    test_layout();
    test_container();
    test_er_isolation();

    printf("checks=%d failed=%d\n", g_checks, g_fail);
    if (g_fail == 0) printf("M0 OK\n");
    return g_fail == 0 ? 0 : 1;
}
