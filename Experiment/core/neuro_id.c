#include "neuro_types.h"
#include <string.h>

static int b36_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'Z') return c - 'A' + 10;
    return -1;
}

static int b62_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'Z') return c - 'A' + 10;
    if (c >= 'a' && c <= 'z') return c - 'a' + 36;
    return -1;
}

static const char kB62[] =
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/* Split "TYPE-BATCH-PARENT-INDIV" into four segments. */
static int split_segments(const char *s, const char **seg, int *len) {
    const char *p = s;
    int n = 0;
    while (n < 4) {
        seg[n] = p;
        while (*p && *p != '-') p++;
        len[n] = (int)(p - seg[n]);
        n++;
        if (*p == '-') {
            if (n == 4) return -1;   /* too many segments */
            p++;
        } else {
            break;
        }
    }
    return (n == 4) ? 0 : -1;
}

int neuro_id_parse(const char *s, NeuroIdParts *out) {
    const char *seg[4];
    int len[4];
    if (!s || !out) return NEURO_E_BAD_ARG;
    if (split_segments(s, seg, len) != 0) return NEURO_E_ID_PARSE;

    if (len[0] < 2 || len[0] > 4) return NEURO_E_ID_PARSE;
    if (len[1] != 4 || len[2] != 4 || len[3] != 6) return NEURO_E_ID_PARSE;

    memset(out, 0, sizeof(*out));
    memcpy(out->type, seg[0], (size_t)len[0]);
    out->type[len[0]] = '\0';

    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        int d = b36_value(seg[1][i]);
        if (d < 0) return NEURO_E_ID_PARSE;
        v = v * 36u + (uint32_t)d;
    }
    out->batch = v;

    v = 0;
    for (int i = 0; i < 4; i++) {
        int d = b36_value(seg[2][i]);
        if (d < 0) return NEURO_E_ID_PARSE;
        v = v * 36u + (uint32_t)d;
    }
    out->parent = v;

    uint64_t w = 0;
    for (int i = 0; i < 6; i++) {
        int d = b62_value(seg[3][i]);
        if (d < 0) return NEURO_E_ID_PARSE;
        w = w * 62u + (uint64_t)d;
    }
    if (w > 0xFFFFFFFFULL) return NEURO_E_ID_RANGE;
    out->indiv = (uint32_t)w;

    return NEURO_OK;
}

int neuro_id_format(const NeuroIdParts *p, char *buf, size_t cap) {
    char tmp[32];
    size_t tn;
    int pos = (int)sizeof(tmp);

    if (!p || !buf) return NEURO_E_BAD_ARG;
    tn = strlen(p->type);
    if (tn < 2 || tn > 4) return NEURO_E_BAD_ARG;

    /* INDIV: 6 base62, zero padded */
    uint32_t indiv = p->indiv;
    for (int i = 0; i < 6; i++) {
        tmp[--pos] = kB62[indiv % 62u];
        indiv /= 62u;
    }
    tmp[--pos] = '-';

    /* PARENT: 4 base36 */
    uint32_t parent = p->parent;
    for (int i = 0; i < 4; i++) {
        unsigned d = parent % 36u;
        tmp[--pos] = (char)(d < 10u ? ('0' + d) : ('A' + (d - 10u)));
        parent /= 36u;
    }
    tmp[--pos] = '-';

    /* BATCH: 4 base36 */
    uint32_t batch = p->batch;
    for (int i = 0; i < 4; i++) {
        unsigned d = batch % 36u;
        tmp[--pos] = (char)(d < 10u ? ('0' + d) : ('A' + (d - 10u)));
        batch /= 36u;
    }
    tmp[--pos] = '-';

    for (int i = (int)tn - 1; i >= 0; i--) tmp[--pos] = p->type[i];

    size_t n = sizeof(tmp) - (size_t)pos;
    if (cap < n + 1) return NEURO_E_BAD_ARG;
    memcpy(buf, tmp + pos, n);
    buf[n] = '\0';
    return NEURO_OK;
}

int neuro_neuron_key_make(const NeuroIdParts *p, NeuronKey *out) {
    char buf[32];
    int rc;
    if (!p || !out) return NEURO_E_BAD_ARG;
    rc = neuro_id_format(p, buf, sizeof(buf));
    if (rc != NEURO_OK) return rc;
    out->id_hash   = neuro_fnv1a64(buf);
    out->type_code = neuro_type_code(p->type);
    out->batch     = p->batch;
    out->parent    = p->parent;
    out->indiv     = p->indiv;
    return NEURO_OK;
}
