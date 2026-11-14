#include "core/png.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---- checksums ---------------------------------------------------------------------------

static uint32_t crc_table[256];
static int crc_ready;

static uint32_t crc32_update(uint32_t c, const unsigned char *p, size_t n){
    if (!crc_ready){
        for (uint32_t i = 0; i < 256; i++){
            uint32_t k = i;
            for (int j = 0; j < 8; j++) k = k & 1 ? 0xEDB88320u ^ (k >> 1) : k >> 1;
            crc_table[i] = k;
        }
        crc_ready = 1;
    }
    c = ~c;
    for (size_t i = 0; i < n; i++) c = crc_table[(c ^ p[i]) & 255] ^ (c >> 8);
    return ~c;
}

static uint32_t adler32(const unsigned char *p, size_t n){
    uint32_t a = 1, b = 0;
    while (n > 0){
        size_t k = n < 5552 ? n : 5552;            // the most bytes before the sums can overflow
        n -= k;
        while (k--){ a += *p++; b += a; }
        a %= 65521;
        b %= 65521;
    }
    return (b << 16) | a;
}

// ---- a growing byte buffer -----------------------------------------------------------------

typedef struct { unsigned char *p; size_t n, cap; int bad; } Buf;

static void buf_need(Buf *b, size_t more){
    if (b->bad || b->n + more <= b->cap) return;
    size_t cap = b->cap ? b->cap : 4096;
    while (cap < b->n + more) cap *= 2;
    unsigned char *q = realloc(b->p, cap);
    if (!q){ b->bad = 1; return; }
    b->p = q;
    b->cap = cap;
}
static void buf_put(Buf *b, const void *d, size_t n){
    buf_need(b, n);
    if (b->bad) return;
    memcpy(b->p + b->n, d, n);
    b->n += n;
}
static void buf_byte(Buf *b, int v){ unsigned char c = (unsigned char)v; buf_put(b, &c, 1); }
static void buf_be32(Buf *b, uint32_t v){
    unsigned char c[4] = {(unsigned char)(v >> 24), (unsigned char)(v >> 16), (unsigned char)(v >> 8), (unsigned char)v};
    buf_put(b, c, 4);
}

// ---- deflate: the tables of RFC 1951 --------------------------------------------------------

static const uint16_t len_base[29] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59,
                                      67, 83, 99, 115, 131, 163, 195, 227, 258};
static const uint8_t len_extra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
static const uint16_t dist_base[30] = {1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769,
                                       1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
static const uint8_t dist_extra[30] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11,
                                       12, 12, 13, 13};

// ---- deflate: compression --------------------------------------------------------------------
//
// LZ77: every position is hashed by its next 3 bytes into chains of earlier positions with
// the same hash; the longest match among the last MAX_CHAIN of them, within 32 KB, replaces
// the bytes by (length, distance). The tokens go out in blocks of BLOCK_TOKENS, each with
// Huffman codes made for its own symbol counts (RFC 1951 3.2.7: "dynamic" codes), or with
// the fixed codes when those come out shorter (a short block).

#define WBITS 15
#define WSIZE (1 << WBITS)
#define HBITS 15
#define MAX_CHAIN 48
#define NICE_LEN 128
#define BLOCK_TOKENS 65536

typedef struct { Buf out; uint32_t bits; int nbits; } BitW;

static void bw_put(BitW *w, uint32_t v, int n){          // n bits of v, the lowest first
    w->bits |= v << w->nbits;
    w->nbits += n;
    while (w->nbits >= 8){
        buf_byte(&w->out, (int)(w->bits & 255));
        w->bits >>= 8;
        w->nbits -= 8;
    }
}
static uint32_t reverse(uint32_t c, int n){
    uint32_t r = 0;
    for (int i = 0; i < n; i++){ r = (r << 1) | (c & 1); c >>= 1; }
    return r;
}

static int len_code(int len){ int i = 28; while (len_base[i] > len) i--; return i; }
static int dist_code(int d){ int j = 29; while (dist_base[j] > d) j--; return j; }

// Code lengths (at most `limit` bits) for symbols with these counts: Huffman's tree built
// with a heap; if it is deeper than the limit, the longest codes are moved up and the
// shortest pushed down until the code is complete again (as zlib and miniz do), and the
// lengths handed out again, the longest to the rarest symbols.
static void huff_lengths(const uint32_t *freq, int n, int limit, uint8_t *len){
    int used[288], nu = 0;
    for (int i = 0; i < n; i++){ len[i] = 0; if (freq[i]) used[nu++] = i; }
    if (nu == 0) return;
    if (nu == 1){ len[used[0]] = 1; return; }
    // the tree: nodes 0..nu-1 are the leaves, then the internal ones
    uint32_t w[576];
    int parent[576], heap[576], hn = 0;
    for (int i = 0; i < nu; i++) w[i] = freq[used[i]];
#define HLESS(a, b) (w[a] < w[b] || (w[a] == w[b] && (a) > (b)))
    for (int i = 0; i < nu; i++){                      // push
        int k = hn++;
        heap[k] = i;
        while (k && HLESS(heap[k], heap[(k - 1) / 2])){ int t = heap[k]; heap[k] = heap[(k - 1) / 2]; heap[(k - 1) / 2] = t; k = (k - 1) / 2; }
    }
    int next = nu;
    while (hn > 1){
        int two[2];
        for (int t = 0; t < 2; t++){                   // pop the lightest twice
            two[t] = heap[0];
            heap[0] = heap[--hn];
            int k = 0;
            for (;;){
                int l = 2 * k + 1, r = l + 1, m = k;
                if (l < hn && HLESS(heap[l], heap[m])) m = l;
                if (r < hn && HLESS(heap[r], heap[m])) m = r;
                if (m == k) break;
                int tmp = heap[k]; heap[k] = heap[m]; heap[m] = tmp;
                k = m;
            }
        }
        w[next] = w[two[0]] + w[two[1]];
        parent[two[0]] = parent[two[1]] = next;
        int k = hn++;
        heap[k] = next;
        while (k && HLESS(heap[k], heap[(k - 1) / 2])){ int t = heap[k]; heap[k] = heap[(k - 1) / 2]; heap[(k - 1) / 2] = t; k = (k - 1) / 2; }
        next++;
    }
#undef HLESS
    int root = next - 1, count[32] = {0};
    int depth[576];
    depth[root] = 0;
    for (int i = root - 1; i >= 0; i--) depth[i] = depth[parent[i]] + 1;   // parents come after children
    for (int i = 0; i < nu; i++) count[depth[i] < 31 ? depth[i] : 31]++;
    // limit the depth: everything deeper goes to `limit`, then repair the Kraft sum
    for (int l = limit + 1; l < 32; l++){ count[limit] += count[l]; count[l] = 0; }
    uint32_t total = 0;
    for (int l = 1; l <= limit; l++) total += (uint32_t)count[l] << (limit - l);
    while (total > (1u << limit)){
        count[limit]--;
        for (int l = limit - 1; l > 0; l--)
            if (count[l]){ count[l]--; count[l + 1] += 2; break; }
        total--;
    }
    // hand out the lengths: rarest symbols first get the longest codes
    for (int i = 1; i < nu; i++){                      // used[] sorted by frequency (insertion sort: n <= 288)
        int s = used[i], j = i - 1;
        while (j >= 0 && (freq[used[j]] > freq[s] || (freq[used[j]] == freq[s] && used[j] > s))){ used[j + 1] = used[j]; j--; }
        used[j + 1] = s;
    }
    int k = 0;
    for (int l = limit; l >= 1; l--)
        for (int c = 0; c < count[l]; c++) len[used[k++]] = (uint8_t)l;
}

// canonical codes from lengths (RFC 1951 3.2.2)
static void huff_codes(const uint8_t *len, int n, uint16_t *code){
    int count[16] = {0}, next[16];
    for (int i = 0; i < n; i++) count[len[i]]++;
    count[0] = 0;
    int c = 0;
    for (int l = 1; l < 16; l++){ c = (c + count[l - 1]) << 1; next[l] = c; }
    for (int i = 0; i < n; i++) if (len[i]) code[i] = (uint16_t)reverse((uint32_t)next[len[i]]++, len[i]);
}

// A token: a literal byte (bit 31 clear) or a match (bit 31, length << 16 | distance).
#define MATCH(len, d) (0x80000000u | (uint32_t)(len) << 16 | (uint32_t)(d))

static void put_block(BitW *w, const uint32_t *tok, size_t nt, int last){
    uint32_t lf[288] = {0}, df[30] = {0};
    for (size_t i = 0; i < nt; i++){
        if (!(tok[i] >> 31)){ lf[tok[i]]++; continue; }
        lf[257 + len_code((int)(tok[i] >> 16 & 0x7FFF))]++;
        df[dist_code((int)(tok[i] & 0xFFFF))]++;
    }
    lf[256] = 1;
    // a code needs two symbols at least (one code of length 1 is incomplete)
    if (lf[0] == 0) lf[0] = 1;
    int nd = 0;
    for (int i = 0; i < 30; i++) nd += df[i] != 0;
    if (nd < 2){ if (!df[0]) df[0] = 1; else df[1] = 1; }
    uint8_t ll[288], dl[30];
    huff_lengths(lf, 286, 15, ll);
    huff_lengths(df, 30, 15, dl);
    ll[286] = ll[287] = 0;
    int hlit = 286, hdist = 30;
    while (hlit > 257 && !ll[hlit - 1]) hlit--;
    while (hdist > 1 && !dl[hdist - 1]) hdist--;
    // the lengths, run-length coded: 16 = repeat the last 3-6 times, 17 / 18 = 3-10 / 11-138 zeros
    uint8_t all[316], rle[316], extra[316];
    int na = 0, nr = 0;
    for (int i = 0; i < hlit; i++) all[na++] = ll[i];
    for (int i = 0; i < hdist; i++) all[na++] = dl[i];
    for (int i = 0; i < na;){
        int v = all[i], run = 1;
        while (i + run < na && all[i + run] == v) run++;
        if (v == 0 && run >= 3){
            int r = run < 138 ? run : 138;
            rle[nr] = r <= 10 ? 17 : 18;
            extra[nr++] = (uint8_t)(r <= 10 ? r - 3 : r - 11);
            i += r;
        } else if (v != 0 && run >= 4){
            rle[nr] = (uint8_t)v; extra[nr++] = 0;
            int r = run - 1 < 6 ? run - 1 : 6;
            rle[nr] = 16; extra[nr++] = (uint8_t)(r - 3);
            i += 1 + r;
        } else { rle[nr] = (uint8_t)v; extra[nr++] = 0; i++; }
    }
    uint32_t cf[19] = {0};
    for (int i = 0; i < nr; i++) cf[rle[i]]++;
    int ncs = 0;                                       // (this code must be complete: two symbols)
    for (int i = 0; i < 19; i++) ncs += cf[i] != 0;
    if (ncs < 2){ if (!cf[0]) cf[0] = 1; else cf[1] = 1; }
    uint8_t cl[19];
    huff_lengths(cf, 19, 7, cl);
    static const uint8_t order[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
    int hclen = 19;
    while (hclen > 4 && !cl[order[hclen - 1]]) hclen--;
    // the size of both ways, in bits: dynamic (with its tables) or fixed
    uint8_t fl[288], fd[30];
    for (int i = 0; i < 288; i++) fl[i] = i < 144 ? 8 : (i < 256 ? 9 : (i < 280 ? 7 : 8));
    for (int i = 0; i < 30; i++) fd[i] = 5;
    uint64_t dyn = 14 + 3 * (uint64_t)hclen, fix = 0;
    for (int i = 0; i < nr; i++) dyn += cl[rle[i]] + (rle[i] == 16 ? 2 : rle[i] == 17 ? 3 : rle[i] == 18 ? 7 : 0);
    for (int i = 0; i < 286; i++){ dyn += (uint64_t)lf[i] * ll[i]; fix += (uint64_t)lf[i] * fl[i]; }
    for (int i = 0; i < 30; i++){ dyn += (uint64_t)df[i] * dl[i]; fix += (uint64_t)df[i] * fd[i]; }
    for (int i = 0; i < 29; i++){ uint64_t e = (uint64_t)lf[257 + i] * len_extra[i]; dyn += e; fix += e; }
    for (int i = 0; i < 30; i++){ uint64_t e = (uint64_t)df[i] * dist_extra[i]; dyn += e; fix += e; }
    const uint8_t *L = ll, *D = dl;
    bw_put(w, (uint32_t)last, 1);
    if (fix <= dyn){
        bw_put(w, 1, 2);
        L = fl;
        D = fd;
    } else {
        bw_put(w, 2, 2);
        bw_put(w, (uint32_t)(hlit - 257), 5);
        bw_put(w, (uint32_t)(hdist - 1), 5);
        bw_put(w, (uint32_t)(hclen - 4), 4);
        for (int i = 0; i < hclen; i++) bw_put(w, cl[order[i]], 3);
        uint16_t cc[19];
        huff_codes(cl, 19, cc);
        for (int i = 0; i < nr; i++){
            bw_put(w, cc[rle[i]], cl[rle[i]]);
            if (rle[i] == 16) bw_put(w, extra[i], 2);
            else if (rle[i] == 17) bw_put(w, extra[i], 3);
            else if (rle[i] == 18) bw_put(w, extra[i], 7);
        }
    }
    uint16_t lc[288], dc[30];
    huff_codes(L, 288, lc);
    huff_codes(D, 30, dc);
    for (size_t i = 0; i < nt; i++){
        uint32_t t = tok[i];
        if (!(t >> 31)){ bw_put(w, lc[t], L[t]); continue; }
        int len = (int)(t >> 16 & 0x7FFF), d = (int)(t & 0xFFFF), a = len_code(len), b = dist_code(d);
        bw_put(w, lc[257 + a], L[257 + a]);
        if (len_extra[a]) bw_put(w, (uint32_t)(len - len_base[a]), len_extra[a]);
        bw_put(w, dc[b], D[b]);
        if (dist_extra[b]) bw_put(w, (uint32_t)(d - dist_base[b]), dist_extra[b]);
    }
    bw_put(w, lc[256], L[256]);                        // the end of the block
}

static uint32_t hash3(const unsigned char *p){
    return ((uint32_t)p[0] << 16 | (uint32_t)p[1] << 8 | p[2]) * 2654435761u >> (32 - HBITS);
}

unsigned char *zlib_compress(const unsigned char *data, size_t n, size_t *out_n){
    BitW w = {{0}, 0, 0};
    buf_need(&w.out, n / 3 + 64);
    buf_byte(&w.out, 0x78);                            // deflate, 32 KB window
    buf_byte(&w.out, 0x01);                            // (0x7801 % 31 == 0), no dictionary
    int32_t *head = malloc(sizeof(int32_t) << HBITS);
    int32_t *prev = malloc(sizeof(int32_t) * WSIZE);
    uint32_t *tok = calloc(BLOCK_TOKENS, sizeof(uint32_t));
    if (!head || !prev || !tok){ free(head); free(prev); free(tok); free(w.out.p); return NULL; }
    for (size_t i = 0; i < ((size_t)1 << HBITS); i++) head[i] = -1;
    size_t i = 0, nt = 0;
    while (i < n){
        int best = 0, best_d = 0;
        if (i + 3 <= n){
            uint32_t h = hash3(data + i);
            int32_t p = head[h];
            int chain = MAX_CHAIN;
            size_t maxlen = n - i < 258 ? n - i : 258;
            while (p >= 0 && i - (size_t)p <= WSIZE && chain-- > 0){
                const unsigned char *a = data + p, *b = data + i;
                if (a[best] == b[best]){                 // can it beat the best? (cheap test first)
                    size_t l = 0;
                    while (l < maxlen && a[l] == b[l]) l++;
                    if ((int)l > best){
                        best = (int)l;
                        best_d = (int)(i - (size_t)p);
                        if (l >= NICE_LEN || l == maxlen) break;
                    }
                }
                int32_t q = prev[p & (WSIZE - 1)];
                if (q >= p) break;                       // overwritten: older than the window
                p = q;
            }
        }
        size_t take = best >= 3 ? (size_t)best : 1;
        tok[nt++] = best >= 3 ? MATCH(best, best_d) : data[i];
        if (nt == BLOCK_TOKENS){ put_block(&w, tok, nt, 0); nt = 0; }
        for (size_t k = 0; k < take; k++, i++){         // every position joins its chain
            if (i + 3 > n) continue;
            uint32_t h = hash3(data + i);
            prev[i & (WSIZE - 1)] = head[h];
            head[h] = (int32_t)i;
        }
    }
    put_block(&w, tok, nt, 1);
    free(head);
    free(prev);
    free(tok);
    if (w.nbits) bw_put(&w, 0, 8 - w.nbits);           // to a whole byte
    buf_be32(&w.out, adler32(data, n));
    if (w.out.bad){ free(w.out.p); return NULL; }
    *out_n = w.out.n;
    return w.out.p;
}

// ---- inflate: decompression ------------------------------------------------------------------

typedef struct {
    const unsigned char *in;
    size_t n, pos;
    uint32_t bits;
    int nbits;
    int over;                                          // bytes read past the end (as zeros)
} BitR;

static void br_fill(BitR *r, int need){
    while (r->nbits < need){
        uint32_t c = 0;
        if (r->pos < r->n) c = r->in[r->pos++];
        else r->over++;
        r->bits |= c << r->nbits;
        r->nbits += 8;
    }
}
static uint32_t br_bits(BitR *r, int n){
    if (n == 0) return 0;
    br_fill(r, n);
    uint32_t v = r->bits & ((1u << n) - 1);
    r->bits >>= n;
    r->nbits -= n;
    return v;
}

// A canonical Huffman code: count[l] codes of length l, symbols sorted by code; a table of
// FAST bits answers most lookups at once (length and symbol), longer codes go bit by bit.
#define FAST 9
typedef struct {
    uint16_t count[16];
    uint16_t sym[288];
    uint16_t fast[1 << FAST];                          // (length << 9 | symbol), 0 = longer code
} Huff;

static int huff_build(Huff *h, const uint8_t *len, int n){
    memset(h, 0, sizeof(*h));
    for (int i = 0; i < n; i++) h->count[len[i]]++;
    h->count[0] = 0;
    int left = 1;                                      // over-subscribed sets are invalid
    for (int l = 1; l < 16; l++){
        left <<= 1;
        left -= h->count[l];
        if (left < 0) return 1;
    }
    uint16_t offs[16];
    offs[1] = 0;
    for (int l = 1; l < 15; l++) offs[l + 1] = offs[l] + h->count[l];
    for (int i = 0; i < n; i++) if (len[i]) h->sym[offs[len[i]]++] = (uint16_t)i;
    // the fast table: the canonical code of each symbol, reversed (the stream is LSB first)
    uint32_t code = 0;
    int k = 0;
    for (int l = 1; l <= FAST; l++){
        for (int c = 0; c < h->count[l]; c++, k++){
            uint32_t r = reverse(code + (uint32_t)c, l);
            for (uint32_t f = r; f < (1u << FAST); f += 1u << l) h->fast[f] = (uint16_t)(l << 9 | h->sym[k]);
        }
        code = (code + h->count[l]) << 1;
    }
    return 0;
}

static int huff_decode(BitR *r, const Huff *h){
    br_fill(r, FAST);
    uint16_t e = h->fast[r->bits & ((1u << FAST) - 1)];
    if (e){
        int l = e >> 9;
        r->bits >>= l;
        r->nbits -= l;
        return e & 511;
    }
    int code = 0, first = 0, index = 0;                // bit by bit (RFC 1951's canonical order)
    for (int l = 1; l < 16; l++){
        code |= (int)br_bits(r, 1);
        int count = h->count[l];
        if (code - first < count) return h->sym[index + code - first];
        index += count;
        first += count;
        first <<= 1;
        code <<= 1;
    }
    return -1;
}

static int inflate_into(BitR *r, Buf *out){
    static Huff fixed_lit, fixed_dist;
    static int fixed_ready;
    if (!fixed_ready){
        uint8_t l[288];
        for (int i = 0; i < 288; i++) l[i] = i < 144 ? 8 : (i < 256 ? 9 : (i < 280 ? 7 : 8));
        huff_build(&fixed_lit, l, 288);
        for (int i = 0; i < 30; i++) l[i] = 5;
        huff_build(&fixed_dist, l, 30);
        fixed_ready = 1;
    }
    int last;
    do {
        last = (int)br_bits(r, 1);
        int type = (int)br_bits(r, 2);
        if (type == 0){                                // stored: to a byte boundary, LEN, NLEN, bytes
            r->bits >>= r->nbits & 7;
            r->nbits -= r->nbits & 7;
            uint32_t len = br_bits(r, 16), nlen = br_bits(r, 16);
            if ((len ^ 0xFFFF) != nlen) return 1;
            buf_need(out, len);
            if (out->bad) return 1;
            while (len > 0 && r->nbits >= 8){ out->p[out->n++] = (unsigned char)br_bits(r, 8); len--; }
            if (r->pos + len > r->n) return 1;
            memcpy(out->p + out->n, r->in + r->pos, len);
            out->n += len;
            r->pos += len;
            continue;
        }
        if (type == 3) return 1;
        Huff dyn_lit, dyn_dist;
        const Huff *lit = &fixed_lit, *dist = &fixed_dist;
        if (type == 2){                                // the code lengths, themselves Huffman coded
            static const uint8_t order[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
            int nlit = (int)br_bits(r, 5) + 257, ndist = (int)br_bits(r, 5) + 1, ncl = (int)br_bits(r, 4) + 4;
            if (nlit > 286 || ndist > 30) return 1;
            uint8_t cl[19] = {0}, lens[320] = {0};
            for (int i = 0; i < ncl; i++) cl[order[i]] = (uint8_t)br_bits(r, 3);
            Huff clh;
            if (huff_build(&clh, cl, 19)) return 1;
            for (int i = 0; i < nlit + ndist;){
                int s = huff_decode(r, &clh);
                if (s < 0) return 1;
                if (s < 16){ lens[i++] = (uint8_t)s; continue; }
                int rep, v = 0;
                if (s == 16){ if (i == 0) return 1; v = lens[i - 1]; rep = 3 + (int)br_bits(r, 2); }
                else if (s == 17) rep = 3 + (int)br_bits(r, 3);
                else rep = 11 + (int)br_bits(r, 7);
                if (i + rep > nlit + ndist) return 1;
                while (rep--) lens[i++] = (uint8_t)v;
            }
            if (lens[256] == 0) return 1;              // no end-of-block code
            if (huff_build(&dyn_lit, lens, nlit) || huff_build(&dyn_dist, lens + nlit, ndist)) return 1;
            lit = &dyn_lit;
            dist = &dyn_dist;
        }
        for (;;){
            int s = huff_decode(r, lit);
            if (s < 0 || r->over > 4) return 1;
            if (s < 256){
                buf_need(out, 1);
                if (out->bad) return 1;
                out->p[out->n++] = (unsigned char)s;
                continue;
            }
            if (s == 256) break;
            s -= 257;
            if (s >= 29) return 1;
            int len = len_base[s] + (int)br_bits(r, len_extra[s]);
            int ds = huff_decode(r, dist);
            if (ds < 0 || ds >= 30) return 1;
            size_t d = dist_base[ds] + br_bits(r, dist_extra[ds]);
            if (d > out->n) return 1;
            buf_need(out, (size_t)len);
            if (out->bad) return 1;
            unsigned char *o = out->p + out->n;
            const unsigned char *from = o - d;
            for (int k = 0; k < len; k++) o[k] = from[k];   // (may overlap: byte by byte)
            out->n += (size_t)len;
        }
    } while (!last);
    return r->over > 4 ? 1 : 0;
}

static unsigned char *zlib_decompress_hint(const unsigned char *data, size_t n, size_t *out_n, size_t hint){
    if (n < 6 || (data[0] & 15) != 8 || ((data[0] << 8) | data[1]) % 31 != 0 || (data[1] & 32)) return NULL;
    BitR r = {data + 2, n - 2, 0, 0, 0, 0};
    Buf out = {0};
    buf_need(&out, hint ? hint : n * 4);
    if (inflate_into(&r, &out) || out.bad){ free(out.p); return NULL; }
    // the Adler-32 of the data, after the deflate stream's last whole byte
    size_t at = r.pos - (size_t)(r.nbits / 8);
    if (at + 4 <= r.n){
        const unsigned char *a = r.in + at;
        uint32_t want = (uint32_t)a[0] << 24 | (uint32_t)a[1] << 16 | (uint32_t)a[2] << 8 | a[3];
        if (want != adler32(out.p, out.n)){ free(out.p); return NULL; }
    }
    *out_n = out.n;
    return out.p;
}

unsigned char *zlib_decompress(const unsigned char *data, size_t n, size_t *out_n){
    return zlib_decompress_hint(data, n, out_n, 0);
}

// ---- PNG: writing ----------------------------------------------------------------------------

static void chunk(Buf *b, const char *type, const unsigned char *d, size_t n){
    buf_be32(b, (uint32_t)n);
    size_t at = b->n;
    buf_put(b, type, 4);
    if (n) buf_put(b, d, n);
    if (b->bad) return;
    buf_be32(b, crc32_update(0, b->p + at, n + 4));
}

static int paeth(int a, int b, int c){
    int p = a + b - c, pa = abs(p - a), pb = abs(p - b), pc = abs(p - c);
    return pa <= pb && pa <= pc ? a : (pb <= pc ? b : c);
}

// one filtered row: f = the filter (0..4), bpp = bytes per pixel, prior = the row above (or zeros)
static void filter_row(unsigned char *dst, const unsigned char *row, const unsigned char *prior, size_t n, int bpp, int f){
    for (size_t i = 0; i < n; i++){
        int a = i >= (size_t)bpp ? row[i - bpp] : 0, b = prior[i], c = i >= (size_t)bpp ? prior[i - bpp] : 0, p;
        switch (f){
        case 0: p = 0; break;
        case 1: p = a; break;
        case 2: p = b; break;
        case 3: p = (a + b) / 2; break;
        default: p = paeth(a, b, c); break;
        }
        dst[i] = (unsigned char)(row[i] - p);
    }
}

unsigned char *png_encode(int w, int h, int channels, const unsigned char *px, size_t *size){
    if (w <= 0 || h <= 0 || channels < 1 || channels > 4) return NULL;
    static const int ctype[5] = {0, 0, 4, 2, 6};
    size_t stride = (size_t)w * channels;
    unsigned char *raw = malloc((stride + 1) * (size_t)h), *zero = calloc(stride, 1), *tmp = malloc(stride);
    if (!raw || !zero || !tmp){ free(raw); free(zero); free(tmp); return NULL; }
    // each row with the filter that makes it smallest by the sum of |byte| (as signed),
    // libpng's heuristic: it leaves the most zeros and small values for the compressor
    for (int y = 0; y < h; y++){
        const unsigned char *row = px + (size_t)y * stride, *prior = y ? px + (size_t)(y - 1) * stride : zero;
        unsigned char *dst = raw + (size_t)y * (stride + 1);
        long best = -1;
        for (int f = 0; f < 5; f++){
            filter_row(tmp, row, prior, stride, channels, f);
            long s = 0;
            for (size_t i = 0; i < stride; i++) s += tmp[i] < 128 ? tmp[i] : 256 - tmp[i];
            if (best < 0 || s < best){ best = s; dst[0] = (unsigned char)f; memcpy(dst + 1, tmp, stride); }
        }
    }
    free(zero);
    free(tmp);
    size_t zn;
    unsigned char *z = zlib_compress(raw, (stride + 1) * (size_t)h, &zn);
    free(raw);
    if (!z) return NULL;
    Buf b = {0};
    static const unsigned char sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    buf_put(&b, sig, 8);
    unsigned char ihdr[13] = {(unsigned char)(w >> 24), (unsigned char)(w >> 16), (unsigned char)(w >> 8), (unsigned char)w,
                              (unsigned char)(h >> 24), (unsigned char)(h >> 16), (unsigned char)(h >> 8), (unsigned char)h,
                              8, (unsigned char)ctype[channels], 0, 0, 0};
    chunk(&b, "IHDR", ihdr, 13);
    chunk(&b, "IDAT", z, zn);
    chunk(&b, "IEND", NULL, 0);
    free(z);
    if (b.bad){ free(b.p); return NULL; }
    *size = b.n;
    return b.p;
}

int png_write(const char *path, int w, int h, int channels, const unsigned char *px){
    size_t n;
    unsigned char *d = png_encode(w, h, channels, px, &n);
    if (!d) return 1;
    FILE *f = fopen(path, "wb");
    if (!f){ free(d); return 1; }
    size_t ok = fwrite(d, 1, n, f);
    free(d);
    return (fclose(f) == 0 && ok == n) ? 0 : 1;
}

// ---- PNG: reading ----------------------------------------------------------------------------

static uint32_t be32(const unsigned char *p){
    return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3];
}

static unsigned char *fail(const char *why){
    fprintf(stderr, "png: %s\n", why);
    return NULL;
}

// sample number i (0-based) of a row of `depth`-bit samples, as the file has it
static unsigned sample(const unsigned char *row, size_t i, int depth){
    if (depth == 8) return row[i];
    if (depth == 16) return (unsigned)row[2 * i] << 8 | row[2 * i + 1];
    size_t bit = i * (size_t)depth;
    return (row[bit >> 3] >> (8 - depth - (int)(bit & 7))) & ((1u << depth) - 1);
}

unsigned char *png_decode(const unsigned char *data, size_t size, int *pw, int *ph, int *pch, int want){
    static const unsigned char sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    if (size < 8 || memcmp(data, sig, 8)) return fail("not a PNG file");
    if (want < 0 || want > 4) return fail("channels must be 0..4");
    uint32_t w = 0, h = 0;
    int depth = 0, ctype = -1, interlace = 0, npal = 0, ntrns = 0, have_key = 0;
    unsigned char pal[256][4];
    unsigned key[3] = {0, 0, 0};
    Buf idat = {0};
    size_t pos = 8;
    int ended = 0;
    while (!ended){
        if (pos + 12 > size){ free(idat.p); return fail("truncated"); }
        uint32_t len = be32(data + pos);
        const unsigned char *type = data + pos + 4, *d = data + pos + 8;
        if (len > size - pos - 12){ free(idat.p); return fail("truncated chunk"); }
        if (crc32_update(0, type, len + 4) != be32(d + len)){ free(idat.p); return fail("bad chunk CRC"); }
        if (!memcmp(type, "IHDR", 4)){
            if (len != 13){ free(idat.p); return fail("bad IHDR"); }
            w = be32(d);
            h = be32(d + 4);
            depth = d[8];
            ctype = d[9];
            interlace = d[12];
            int ok = (ctype == 0 && (depth == 1 || depth == 2 || depth == 4 || depth == 8 || depth == 16)) ||
                     (ctype == 3 && (depth == 1 || depth == 2 || depth == 4 || depth == 8)) ||
                     ((ctype == 2 || ctype == 4 || ctype == 6) && (depth == 8 || depth == 16));
            if (!ok || d[10] || d[11] || interlace > 1 || w == 0 || h == 0 || w > 1u << 24 || h > 1u << 24){
                free(idat.p);
                return fail("unsupported IHDR");
            }
        } else if (!memcmp(type, "PLTE", 4)){
            npal = (int)(len / 3);
            if (npal > 256 || len % 3){ free(idat.p); return fail("bad palette"); }
            for (int i = 0; i < npal; i++){ pal[i][0] = d[3 * i]; pal[i][1] = d[3 * i + 1]; pal[i][2] = d[3 * i + 2]; pal[i][3] = 255; }
        } else if (!memcmp(type, "tRNS", 4)){
            if (ctype == 3){ ntrns = (int)(len < 256 ? len : 256); for (int i = 0; i < ntrns; i++) pal[i][3] = d[i]; }
            else if (ctype == 0 && len >= 2){ key[0] = (unsigned)d[0] << 8 | d[1]; have_key = 1; }
            else if (ctype == 2 && len >= 6){
                for (int k = 0; k < 3; k++) key[k] = (unsigned)d[2 * k] << 8 | d[2 * k + 1];
                have_key = 1;
            }
        } else if (!memcmp(type, "IDAT", 4)){
            buf_put(&idat, d, len);
        } else if (!memcmp(type, "IEND", 4)){
            ended = 1;
        } else if (!(type[0] & 32)){                    // an unknown critical chunk
            free(idat.p);
            return fail("unknown critical chunk");
        }
        pos += 12 + len;
    }
    if (ctype < 0 || !idat.n || idat.bad){ free(idat.p); return fail("no image data"); }
    if (ctype == 3 && npal == 0){ free(idat.p); return fail("palette missing"); }

    const int spp = ctype == 0 ? 1 : ctype == 2 ? 3 : ctype == 3 ? 1 : ctype == 4 ? 2 : 4;
    const int bits = spp * depth, bpp = bits >= 8 ? bits / 8 : 1;
    // the bytes the passes need (the filter byte of each row included)
    static const int xs[7] = {0, 4, 0, 2, 0, 1, 0}, ys[7] = {0, 0, 4, 0, 2, 0, 1},
                     dx[7] = {8, 8, 4, 4, 2, 2, 1}, dy[7] = {8, 8, 8, 4, 4, 2, 2};
    int npass = interlace ? 7 : 1;
    size_t need = 0;
    for (int p = 0; p < npass; p++){
        size_t pwid = interlace ? (w - xs[p] + dx[p] - 1) / dx[p] : w, phgt = interlace ? (h - ys[p] + dy[p] - 1) / dy[p] : h;
        if (xs[p] >= (int)w) pwid = 0;
        if (ys[p] >= (int)h) phgt = 0;
        if (pwid && phgt) need += phgt * ((pwid * bits + 7) / 8 + 1);
    }
    size_t rn;
    unsigned char *raw = zlib_decompress_hint(idat.p, idat.n, &rn, need);
    free(idat.p);
    if (!raw) return fail("bad compressed data");
    if (rn < need){ free(raw); return fail("image data too short"); }

    unsigned char *rgba = malloc((size_t)w * h * 4);
    if (!rgba){ free(raw); return fail("out of memory"); }
    unsigned char *at = raw;
    for (int p = 0; p < npass; p++){
        size_t pwid = interlace ? (w - xs[p] + dx[p] - 1) / dx[p] : w, phgt = interlace ? (h - ys[p] + dy[p] - 1) / dy[p] : h;
        if (interlace && (xs[p] >= (int)w || ys[p] >= (int)h)) continue;
        if (!pwid || !phgt) continue;
        size_t stride = (pwid * bits + 7) / 8;
        unsigned char *prior = NULL;
        for (size_t y = 0; y < phgt; y++){
            int f = at[0];
            unsigned char *row = at + 1;
            if (f > 4){ free(raw); free(rgba); return fail("bad filter"); }
            for (size_t i = 0; i < stride; i++){          // undo the filter, in place
                int a = i >= (size_t)bpp ? row[i - bpp] : 0, b = prior ? prior[i] : 0,
                    c = (prior && i >= (size_t)bpp) ? prior[i - bpp] : 0, pr;
                switch (f){
                case 0: pr = 0; break;
                case 1: pr = a; break;
                case 2: pr = b; break;
                case 3: pr = (a + b) / 2; break;
                default: pr = paeth(a, b, c); break;
                }
                row[i] = (unsigned char)(row[i] + pr);
            }
            size_t oy = interlace ? ys[p] + y * dy[p] : y;
            for (size_t x = 0; x < pwid; x++){
                size_t ox = interlace ? xs[p] + x * dx[p] : x;
                unsigned char *o = rgba + (oy * w + ox) * 4;
                unsigned v[4];
                for (int k = 0; k < spp; k++) v[k] = sample(row, x * spp + k, depth);
                if (ctype == 3){
                    if (v[0] >= (unsigned)npal){ free(raw); free(rgba); return fail("palette index out of range"); }
                    memcpy(o, pal[v[0]], 4);
                    continue;
                }
                int alpha = 255;
                if (have_key && (ctype == 0 ? v[0] == key[0] : (v[0] == key[0] && v[1] == key[1] && v[2] == key[2]))) alpha = 0;
                for (int k = 0; k < spp; k++)
                    v[k] = depth == 16 ? v[k] >> 8 : depth == 8 ? v[k] : v[k] * 255 / ((1u << depth) - 1);
                if (ctype == 0){ o[0] = o[1] = o[2] = (unsigned char)v[0]; o[3] = (unsigned char)alpha; }
                else if (ctype == 4){ o[0] = o[1] = o[2] = (unsigned char)v[0]; o[3] = (unsigned char)v[1]; }
                else { o[0] = (unsigned char)v[0]; o[1] = (unsigned char)v[1]; o[2] = (unsigned char)v[2];
                       o[3] = ctype == 6 ? (unsigned char)v[3] : (unsigned char)alpha; }
            }
            prior = row;
            at += stride + 1;
        }
    }
    free(raw);

    // the file's own channels (a palette is RGB; a tRNS chunk adds alpha)
    int grey = ctype == 0 || ctype == 4;
    int has_alpha = ctype == 4 || ctype == 6 || (ctype == 3 && ntrns > 0) || have_key;
    int file_ch = grey ? (has_alpha ? 2 : 1) : (has_alpha ? 4 : 3);
    int ch = want ? want : file_ch;
    size_t np = (size_t)w * h;
    unsigned char *out = ch == 4 ? rgba : malloc(np * (size_t)ch);
    if (!out){ free(rgba); return fail("out of memory"); }
    if (ch != 4)
        for (size_t i = 0; i < np; i++){
            const unsigned char *s = rgba + 4 * i;
            unsigned char *o = out + (size_t)ch * i;
            int g = grey ? s[0] : (s[0] * 299 + s[1] * 587 + s[2] * 114 + 500) / 1000;
            if (ch == 1) o[0] = (unsigned char)g;
            else if (ch == 2){ o[0] = (unsigned char)g; o[1] = s[3]; }
            else { o[0] = s[0]; o[1] = s[1]; o[2] = s[2]; }
        }
    if (out != rgba) free(rgba);
    *pw = (int)w;
    *ph = (int)h;
    *pch = ch;
    return out;
}

unsigned char *png_read(const char *path, int *w, int *h, int *channels, int want){
    FILE *f = fopen(path, "rb");
    if (!f){ fprintf(stderr, "png: cannot open %s\n", path); return NULL; }
    Buf b = {0};
    unsigned char tmp[65536];
    size_t k;
    while ((k = fread(tmp, 1, sizeof(tmp), f)) > 0) buf_put(&b, tmp, k);
    fclose(f);
    if (b.bad){ free(b.p); return fail("out of memory"); }
    unsigned char *px = png_decode(b.p, b.n, w, h, channels, want);
    free(b.p);
    return px;
}
