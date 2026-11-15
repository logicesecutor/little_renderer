// jpeg.c - see jpeg.h. The steps, in the order of ITU T.81: markers (tables, the frame, the
// scans), entropy decoding of each scan into the components' DCT coefficients (all of them
// kept, as a progressive file refines them scan by scan), then per 8 x 8 block dequantization
// and the inverse DCT (libjpeg's accurate integer one, jidctint.c), chroma upsampling and
// colour conversion.
#include "core/jpeg.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// the zig-zag order: the k-th coefficient of a scan is natural position ZZ[k]
static const uint8_t ZZ[64 + 16] = {
     0,  1,  8, 16,  9,  2,  3, 10, 17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
    63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63    // (a bad run lands here)
};

// ---- Huffman tables --------------------------------------------------------------------------

#define FAST_BITS 9
typedef struct {
    uint8_t fast_len[1 << FAST_BITS];   // codes of up to FAST_BITS bits: length (0 = longer)
    uint8_t fast_sym[1 << FAST_BITS];   // ... and symbol
    int32_t maxcode[18], mincode[17], valptr[17];
    uint8_t vals[256];
    int ok;
} Huff;

static int huff_build(Huff *h, const uint8_t counts[16], const uint8_t *vals, int nvals){
    memset(h, 0, sizeof(*h));
    memcpy(h->vals, vals, (size_t)nvals);
    int code = 0, k = 0;
    for (int len = 1; len <= 16; len++){
        int n = counts[len - 1];
        h->valptr[len] = k;
        h->mincode[len] = code;
        for (int i = 0; i < n; i++, k++, code++){
            if (code >= (1 << len)) return 1;                 // more codes than fit
            if (len <= FAST_BITS){
                int first = code << (FAST_BITS - len), cnt = 1 << (FAST_BITS - len);
                for (int j = 0; j < cnt; j++){ h->fast_len[first + j] = (uint8_t)len; h->fast_sym[first + j] = vals[k]; }
            }
        }
        h->maxcode[len] = n ? code - 1 : -1;
        code <<= 1;
    }
    h->maxcode[17] = 0x7fffffff;
    h->ok = 1;
    return 0;
}

// ---- the bit reader --------------------------------------------------------------------------
// Entropy-coded data: a 0xFF byte is followed by a stuffed 0x00; any other byte after 0xFF is a
// marker (RSTn, or the next segment), which ends the data: zeros are read from there on.

typedef struct {
    const uint8_t *p, *end;
    uint32_t buf;          // the next bits, from the top
    int cnt;               // how many
    int marker;            // a marker found in the data (its second byte), 0 = none yet
} Bits;

static void bits_fill(Bits *b){
    while (b->cnt <= 24){
        unsigned c = 0;
        if (!b->marker && b->p < b->end){
            c = *b->p;
            if (c == 0xFF){
                unsigned c2 = b->p + 1 < b->end ? b->p[1] : 0xD9;
                while (c2 == 0xFF && b->p + 2 < b->end){ b->p++; c2 = b->p[1]; }   // fill bytes
                if (c2 == 0) b->p += 2;
                else { b->marker = (int)c2; c = 0; }
            } else b->p++;
        }
        b->buf |= (uint32_t)c << (24 - b->cnt);
        b->cnt += 8;
    }
}
static inline int bits_get(Bits *b, int n){        // n in 1..16
    if (b->cnt < n) bits_fill(b);
    int v = (int)(b->buf >> (32 - n));
    b->buf <<= n;
    b->cnt -= n;
    return v;
}
static inline int bits_bit(Bits *b){ return bits_get(b, 1); }
static inline int extend(int v, int s){ return v < (1 << (s - 1)) ? v - (1 << s) + 1 : v; }

static int huff_decode(Bits *b, const Huff *h, int *bad){
    if (b->cnt < 16) bits_fill(b);
    unsigned top = b->buf >> (32 - FAST_BITS);
    int len = h->fast_len[top];
    if (len){
        b->buf <<= len;
        b->cnt -= len;
        return h->fast_sym[top];
    }
    unsigned c16 = b->buf >> 16;
    for (len = FAST_BITS + 1; len <= 16; len++){
        int code = (int)(c16 >> (16 - len));
        if (code <= h->maxcode[len]){
            b->buf <<= len;
            b->cnt -= len;
            int i = h->valptr[len] + code - h->mincode[len];
            return i < 256 ? h->vals[i] : 0;
        }
    }
    *bad = 1;                                      // no such code: corrupt data
    b->buf <<= 16;
    b->cnt -= 16;
    return 0;
}

// ---- the decoder's state ---------------------------------------------------------------------

typedef struct {
    int id, h, v, tq;      // sampling factors, quantization table
    int cw, ch;            // size in samples (the image's, scaled by h / hmax, v / vmax)
    int bw, bh;            // blocks across and down, a whole number of MCUs
    int16_t *coef;         // bw x bh blocks of 64, natural order, not yet dequantized
    int dc, td, ta;        // DC prediction; the scan's Huffman tables
    uint8_t *pix;          // after the IDCT: bw * 8 wide, bh * 8 high
} Comp;

typedef struct {
    const uint8_t *data, *end;
    uint16_t q[4][64];     // natural order
    Huff dc[4], ac[4];
    Comp comp[4];
    int ncomp, w, h, hmax, vmax, mcux, mcuy, progressive, frame, restart, adobe, transform;
    int eobrun, bad;
    const char *err;
} Jpeg;

static int be16(const uint8_t *p){ return p[0] << 8 | p[1]; }

// ---- decoding one block ----------------------------------------------------------------------

// the next DC difference, added to the prediction. An 8-bit JPEG's DC categories are 0..11
// (the bit reader takes up to 16): a larger symbol is corrupt data, and the prediction is
// held to a range no valid file leaves, so that nothing overflows on a crafted one
static int dc_diff(Jpeg *j, Bits *b, Comp *c){
    int t = huff_decode(b, &j->dc[c->td], &j->bad);
    if (t > 11){ j->bad = 1; t = 0; }
    int diff = t ? extend(bits_get(b, t), t) : 0;
    c->dc += diff;
    if (c->dc > 65535) c->dc = 65535;
    if (c->dc < -65536) c->dc = -65536;
    return diff;
}

static void block_baseline(Jpeg *j, Bits *b, Comp *c, int16_t *blk){
    dc_diff(j, b, c);
    blk[0] = (int16_t)c->dc;
    const Huff *ac = &j->ac[c->ta];
    for (int k = 1; k < 64;){
        int rs = huff_decode(b, ac, &j->bad);
        int r = rs >> 4, s = rs & 15;
        if (!s){
            if (r != 15) break;                    // end of block
            k += 16;                               // 16 zeros
            continue;
        }
        k += r;
        if (k > 63){ j->bad = 1; break; }
        blk[ZZ[k++]] = (int16_t)extend(bits_get(b, s), s);
    }
}

static void block_dc_first(Jpeg *j, Bits *b, Comp *c, int16_t *blk, int al){
    dc_diff(j, b, c);
    blk[0] = (int16_t)(c->dc * (1 << al));
}

static void block_dc_refine(Bits *b, int16_t *blk, int al){
    if (bits_bit(b)) blk[0] = (int16_t)(blk[0] | (1 << al));
}

static void block_ac_first(Jpeg *j, Bits *b, Comp *c, int16_t *blk, int ss, int se, int al){
    if (j->eobrun){ j->eobrun--; return; }
    const Huff *ac = &j->ac[c->ta];
    for (int k = ss; k <= se;){
        int rs = huff_decode(b, ac, &j->bad);
        int r = rs >> 4, s = rs & 15;
        if (!s){
            if (r < 15){                           // EOBn: this block and 2^r - 1 + n more end here
                j->eobrun = (1 << r) - 1;
                if (r) j->eobrun += bits_get(b, r);
                break;
            }
            k += 16;
            continue;
        }
        k += r;
        if (k > 63){ j->bad = 1; break; }
        blk[ZZ[k++]] = (int16_t)(extend(bits_get(b, s), s) * (1 << al));
    }
}

// a refinement: one more bit for coefficients already non-zero, new ones (+-1 << al) placed
// among the zeros (T.81 G.1.2.3)
static void block_ac_refine(Jpeg *j, Bits *b, Comp *c, int16_t *blk, int ss, int se, int al){
    int p1 = 1 << al, m1 = -(1 << al);
    int k = ss;
    if (!j->eobrun){
        const Huff *ac = &j->ac[c->ta];
        while (k <= se){
            int rs = huff_decode(b, ac, &j->bad);
            int r = rs >> 4, s = rs & 15, v = 0;
            if (!s){
                if (r < 15){
                    j->eobrun = 1 << r;
                    if (r) j->eobrun += bits_get(b, r);
                    break;                         // the rest of the band: below, as an EOB run
                }
            } else {
                if (s != 1) j->bad = 1;
                v = bits_bit(b) ? p1 : m1;
            }
            while (k <= se){
                int16_t *z = &blk[ZZ[k++]];
                if (*z){
                    if (bits_bit(b) && (*z & p1) == 0) *z = (int16_t)(*z + (*z >= 0 ? p1 : m1));
                } else {
                    if (r == 0){
                        if (v) *z = (int16_t)v;
                        break;
                    }
                    r--;
                }
            }
        }
        if (k > se) return;
    }
    // inside an EOB run: only the correction bits of the non-zero ones
    for (; k <= se; k++){
        int16_t *z = &blk[ZZ[k]];
        if (*z && bits_bit(b) && (*z & p1) == 0) *z = (int16_t)(*z + (*z >= 0 ? p1 : m1));
    }
    j->eobrun--;
}

// ---- a scan ------------------------------------------------------------------------------------

// the next marker at or after p (its 0xFF), or end
static const uint8_t *next_marker(const uint8_t *p, const uint8_t *end){
    while (p + 1 < end){
        if (p[0] == 0xFF && p[1] != 0 && p[1] != 0xFF && !(p[1] >= 0xD0 && p[1] <= 0xD7)) return p;
        p++;
    }
    return end;
}

static const uint8_t *scan(Jpeg *j, const uint8_t *p, int len){
    // the header: Ns, then 2 bytes per component, then Ss, Se, Ah/Al (len counts the
    // length's own 2 bytes too)
    const uint8_t *seg = p;
    if (len < 3){ j->err = "bad scan header"; return NULL; }
    int ns = seg[0];
    if (ns < 1 || ns > 4 || len < 2 + 1 + 2 * ns + 3){ j->err = "bad scan header"; return NULL; }
    Comp *sc[4];
    for (int i = 0; i < ns; i++){
        int id = seg[1 + 2 * i], t = seg[2 + 2 * i];
        sc[i] = NULL;
        for (int k = 0; k < j->ncomp; k++) if (j->comp[k].id == id) sc[i] = &j->comp[k];
        if (!sc[i]){ j->err = "scan of an unknown component"; return NULL; }
        sc[i]->td = t >> 4 & 3;
        sc[i]->ta = t & 3;
    }
    int ss = seg[1 + 2 * ns], se = seg[2 + 2 * ns], ah = seg[3 + 2 * ns] >> 4, al = seg[3 + 2 * ns] & 15;
    if (!j->progressive){ ss = 0; se = 63; ah = al = 0; }
    if (ss > se || se > 63 || (ss == 0 && se != 0 && j->progressive) || (ss > 0 && ns != 1) || ah > 13 || al > 13){
        j->err = "bad progressive scan";
        return NULL;
    }
    for (int i = 0; i < ns; i++){
        sc[i]->dc = 0;
        int need_dc = ss == 0 && ah == 0, need_ac = se > 0;
        if ((need_dc && !j->dc[sc[i]->td].ok) || (need_ac && !j->ac[sc[i]->ta].ok)){
            j->err = "a Huffman table is missing";
            return NULL;
        }
    }
    // the entropy-coded data follows the header (len counts the length's own 2 bytes)
    Bits b = {p + len - 2, j->end, 0, 0, 0};
    j->eobrun = 0;
    // the MCUs: with one component, its blocks one by one (only those inside its area);
    // with several, each MCU holds h x v blocks of each
    int units_x, units_y;
    if (ns == 1){
        units_x = (sc[0]->cw + 7) / 8;
        units_y = (sc[0]->ch + 7) / 8;
    } else {
        units_x = j->mcux;
        units_y = j->mcuy;
    }
    long total = (long)units_x * units_y, left = j->restart ? j->restart : total + 1;
    for (long u = 0; u < total; u++){
        if (left == 0){
            // a restart: the bits to the byte boundary dropped, the marker passed over, the
            // predictions reset
            b.buf = 0; b.cnt = 0;
            if (!b.marker){
                while (b.p + 1 < b.end && !(b.p[0] == 0xFF && b.p[1] >= 0xD0 && b.p[1] <= 0xD7)) b.p++;
                if (b.p + 1 < b.end) b.marker = b.p[1];
            }
            if (b.marker >= 0xD0 && b.marker <= 0xD7){ b.p += 2; b.marker = 0; }
            for (int i = 0; i < ns; i++) sc[i]->dc = 0;
            j->eobrun = 0;
            left = j->restart;
        }
        left--;
        int ux = (int)(u % units_x), uy = (int)(u / units_x);
        for (int i = 0; i < ns; i++){
            Comp *c = sc[i];
            int nh = ns == 1 ? 1 : c->h, nv = ns == 1 ? 1 : c->v;
            for (int by = 0; by < nv; by++)
                for (int bx = 0; bx < nh; bx++){
                    int x = ux * nh + bx, y = uy * nv + by;
                    int16_t *blk = c->coef + ((size_t)y * c->bw + x) * 64;
                    if (!j->progressive) block_baseline(j, &b, c, blk);
                    else if (ss == 0){
                        if (ah == 0) block_dc_first(j, &b, c, blk, al);
                        else block_dc_refine(&b, blk, al);
                    } else if (ah == 0) block_ac_first(j, &b, c, blk, ss, se, al);
                    else block_ac_refine(j, &b, c, blk, ss, se, al);
                }
        }
        if (j->bad){ j->err = "corrupt entropy-coded data"; return NULL; }
    }
    return next_marker(b.p, j->end);
}

// ---- the inverse DCT: libjpeg's jidctint.c (Loeffler, Ligtenberg and Moschytz), integer ----

#define CONST_BITS 13
#define PASS1_BITS 2
#define FIX(x) ((int32_t)((x) * (1 << CONST_BITS) + 0.5))
#define DESCALE(x, n) (((x) + ((int32_t)1 << ((n) - 1))) >> (n))

static inline uint8_t clamp8(int v){ return (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v); }

static void idct_block(const int16_t *in, const uint16_t *q, uint8_t *out, int stride){
    int32_t ws[64];
    for (int c = 0; c < 8; c++){
        const int16_t *ip = in + c;
        const uint16_t *qp = q + c;
        int32_t *wp = ws + c;
        if (!ip[8] && !ip[16] && !ip[24] && !ip[32] && !ip[40] && !ip[48] && !ip[56]){
            int32_t dc = (int32_t)ip[0] * qp[0] * (1 << PASS1_BITS);
            for (int r = 0; r < 8; r++) wp[8 * r] = dc;
            continue;
        }
        int32_t z1, z2, z3, z4, z5, t0, t1, t2, t3, t10, t11, t12, t13;
        z2 = ip[16] * qp[16]; z3 = ip[48] * qp[48];
        z1 = (z2 + z3) * FIX(0.541196100);
        t2 = z1 + z3 * -FIX(1.847759065);
        t3 = z1 + z2 * FIX(0.765366865);
        z2 = ip[0] * qp[0]; z3 = ip[32] * qp[32];
        t0 = (z2 + z3) * (1 << CONST_BITS);
        t1 = (z2 - z3) * (1 << CONST_BITS);
        t10 = t0 + t3; t13 = t0 - t3; t11 = t1 + t2; t12 = t1 - t2;
        t0 = ip[56] * qp[56]; t1 = ip[40] * qp[40]; t2 = ip[24] * qp[24]; t3 = ip[8] * qp[8];
        z1 = t0 + t3; z2 = t1 + t2; z3 = t0 + t2; z4 = t1 + t3;
        z5 = (z3 + z4) * FIX(1.175875602);
        t0 *= FIX(0.298631336); t1 *= FIX(2.053119869); t2 *= FIX(3.072711026); t3 *= FIX(1.501321110);
        z1 *= -FIX(0.899976223); z2 *= -FIX(2.562915447); z3 *= -FIX(1.961570560); z4 *= -FIX(0.390180644);
        z3 += z5; z4 += z5;
        t0 += z1 + z3; t1 += z2 + z4; t2 += z2 + z3; t3 += z1 + z4;
        wp[0]  = DESCALE(t10 + t3, CONST_BITS - PASS1_BITS);
        wp[56] = DESCALE(t10 - t3, CONST_BITS - PASS1_BITS);
        wp[8]  = DESCALE(t11 + t2, CONST_BITS - PASS1_BITS);
        wp[48] = DESCALE(t11 - t2, CONST_BITS - PASS1_BITS);
        wp[16] = DESCALE(t12 + t1, CONST_BITS - PASS1_BITS);
        wp[40] = DESCALE(t12 - t1, CONST_BITS - PASS1_BITS);
        wp[24] = DESCALE(t13 + t0, CONST_BITS - PASS1_BITS);
        wp[32] = DESCALE(t13 - t0, CONST_BITS - PASS1_BITS);
    }
    for (int r = 0; r < 8; r++){
        const int32_t *wp = ws + 8 * r;
        uint8_t *op = out + (size_t)r * stride;
        int32_t z1, z2, z3, z4, z5, t0, t1, t2, t3, t10, t11, t12, t13;
        z2 = wp[2]; z3 = wp[6];
        z1 = (z2 + z3) * FIX(0.541196100);
        t2 = z1 + z3 * -FIX(1.847759065);
        t3 = z1 + z2 * FIX(0.765366865);
        t0 = (wp[0] + wp[4]) * (1 << CONST_BITS);
        t1 = (wp[0] - wp[4]) * (1 << CONST_BITS);
        t10 = t0 + t3; t13 = t0 - t3; t11 = t1 + t2; t12 = t1 - t2;
        t0 = wp[7]; t1 = wp[5]; t2 = wp[3]; t3 = wp[1];
        z1 = t0 + t3; z2 = t1 + t2; z3 = t0 + t2; z4 = t1 + t3;
        z5 = (z3 + z4) * FIX(1.175875602);
        t0 *= FIX(0.298631336); t1 *= FIX(2.053119869); t2 *= FIX(3.072711026); t3 *= FIX(1.501321110);
        z1 *= -FIX(0.899976223); z2 *= -FIX(2.562915447); z3 *= -FIX(1.961570560); z4 *= -FIX(0.390180644);
        z3 += z5; z4 += z5;
        t0 += z1 + z3; t1 += z2 + z4; t2 += z2 + z3; t3 += z1 + z4;
        const int n = CONST_BITS + PASS1_BITS + 3;
        op[0] = clamp8(DESCALE(t10 + t3, n) + 128);
        op[7] = clamp8(DESCALE(t10 - t3, n) + 128);
        op[1] = clamp8(DESCALE(t11 + t2, n) + 128);
        op[6] = clamp8(DESCALE(t11 - t2, n) + 128);
        op[2] = clamp8(DESCALE(t12 + t1, n) + 128);
        op[5] = clamp8(DESCALE(t12 - t1, n) + 128);
        op[3] = clamp8(DESCALE(t13 + t0, n) + 128);
        op[4] = clamp8(DESCALE(t13 - t0, n) + 128);
    }
}

// ---- upsampling (libjpeg's jdsample.c) and colour --------------------------------------------

// component c's samples at full size: w x h
static void upsample(const Comp *c, int hmax, int vmax, int w, int h, uint8_t *out){
    int pw = c->bw * 8, cw = c->cw, ch = c->ch;
    int fx = hmax / c->h, fy = vmax / c->v;
    int exact = hmax % c->h == 0 && vmax % c->v == 0;
    for (int y = 0; y < h; y++){
        uint8_t *o = out + (size_t)y * w;
        if (exact && fx == 1 && fy == 1){ memcpy(o, c->pix + (size_t)y * pw, (size_t)w); continue; }
        if (exact && (fx == 2 || fx == 1) && (fy == 2 || fy == 1) && (fx == 1 || cw > 2)){
            // the "fancy" triangle filter: each output sample 3/4 its nearest input sample and
            // 1/4 the next one (both ways for 2 x 2), libjpeg's rounding
            int iy = y / fy;
            const uint8_t *r0 = c->pix + (size_t)iy * pw, *r1 = r0;
            if (fy == 2){
                int ny = y % 2 == 0 ? iy - 1 : iy + 1;
                if (ny < 0) ny = 0;
                if (ny > ch - 1) ny = ch - 1;
                r1 = c->pix + (size_t)ny * pw;
            }
            if (fx == 1){                                    // 1 x 2
                int bias = y % 2 == 0 ? 1 : 2;
                for (int x = 0; x < w; x++) o[x] = (uint8_t)((r0[x] * 3 + r1[x] + bias) >> 2);
                continue;
            }
            uint8_t tmp[2];
            if (fy == 1){                                    // 2 x 1
                for (int x = 0; x < cw; x++){
                    int cur = r0[x] * 3;
                    int a = x == 0 ? r0[0] : (cur + r0[x - 1] + 1) >> 2;
                    int bb = x == cw - 1 ? r0[cw - 1] : (cur + r0[x + 1] + 2) >> 2;
                    tmp[0] = (uint8_t)a; tmp[1] = (uint8_t)bb;
                    if (2 * x < w) o[2 * x] = tmp[0];
                    if (2 * x + 1 < w) o[2 * x + 1] = tmp[1];
                }
                continue;
            }
            for (int x = 0; x < cw; x++){                    // 2 x 2
                int s = r0[x] * 3 + r1[x];
                int sl = x > 0 ? r0[x - 1] * 3 + r1[x - 1] : 0, sr = x < cw - 1 ? r0[x + 1] * 3 + r1[x + 1] : 0;
                int a = x == 0 ? (s * 4 + 8) >> 4 : (s * 3 + sl + 8) >> 4;
                int bb = x == cw - 1 ? (s * 4 + 7) >> 4 : (s * 3 + sr + 7) >> 4;
                if (2 * x < w) o[2 * x] = (uint8_t)a;
                if (2 * x + 1 < w) o[2 * x + 1] = (uint8_t)bb;
            }
            continue;
        }
        // any other factors: the nearest sample
        int iy = (int)((long)y * c->v / vmax);
        if (iy > ch - 1) iy = ch - 1;
        const uint8_t *r = c->pix + (size_t)iy * pw;
        for (int x = 0; x < w; x++){
            int ix = (int)((long)x * c->h / hmax);
            o[x] = r[ix > cw - 1 ? cw - 1 : ix];
        }
    }
}

#define CSCALE 16
#define CFIX(x) ((int32_t)((x) * (1 << CSCALE) + 0.5))

static void ycc_to_rgb(int y, int cb, int cr, uint8_t *o){
    int x = cr - 128, z = cb - 128;
    int r = y + ((CFIX(1.40200) * x + (1 << (CSCALE - 1))) >> CSCALE);
    int b = y + ((CFIX(1.77200) * z + (1 << (CSCALE - 1))) >> CSCALE);
    int g = y + ((-CFIX(0.34414) * z + (1 << (CSCALE - 1)) - CFIX(0.71414) * x) >> CSCALE);
    o[0] = clamp8(r); o[1] = clamp8(g); o[2] = clamp8(b);
}

// ---- the file ----------------------------------------------------------------------------------

static unsigned char *fail(Jpeg *j, const char *why){
    fprintf(stderr, "jpeg: %s\n", why);
    for (int i = 0; i < 4; i++){ free(j->comp[i].coef); free(j->comp[i].pix); }
    return NULL;
}

static int frame(Jpeg *j, const uint8_t *s, int len, int marker){
    if (j->frame){ j->err = "two frames"; return 1; }
    if (len < 6){ j->err = "bad frame header"; return 1; }
    if (s[0] != 8){ j->err = "only 8-bit samples are supported"; return 1; }
    j->h = be16(s + 1); j->w = be16(s + 3); j->ncomp = s[5];
    if (j->w <= 0 || j->h <= 0){ j->err = "no size (a DNL marker is not supported)"; return 1; }
    if ((j->ncomp != 1 && j->ncomp != 3) || len < 6 + 3 * j->ncomp){ j->err = "only grey or 3-component images are supported"; return 1; }
    j->progressive = marker == 0xC2;
    j->hmax = j->vmax = 1;
    for (int i = 0; i < j->ncomp; i++){
        Comp *c = &j->comp[i];
        c->id = s[6 + 3 * i]; c->h = s[7 + 3 * i] >> 4; c->v = s[7 + 3 * i] & 15; c->tq = s[8 + 3 * i] & 3;
        if (c->h < 1 || c->h > 4 || c->v < 1 || c->v > 4){ j->err = "bad sampling factors"; return 1; }
        if (c->h > j->hmax) j->hmax = c->h;
        if (c->v > j->vmax) j->vmax = c->v;
    }
    if (j->ncomp == 1) j->comp[0].h = j->comp[0].v = j->hmax = j->vmax = 1;   // (one component: no MCUs)
    j->mcux = (j->w + 8 * j->hmax - 1) / (8 * j->hmax);
    j->mcuy = (j->h + 8 * j->vmax - 1) / (8 * j->vmax);
    if ((double)j->mcux * j->mcuy * 64 * j->hmax * j->vmax * j->ncomp > 4e8){ j->err = "image too large"; return 1; }
    for (int i = 0; i < j->ncomp; i++){
        Comp *c = &j->comp[i];
        c->cw = (j->w * c->h + j->hmax - 1) / j->hmax;
        c->ch = (j->h * c->v + j->vmax - 1) / j->vmax;
        c->bw = j->mcux * c->h;
        c->bh = j->mcuy * c->v;
        c->coef = calloc((size_t)c->bw * c->bh * 64, sizeof(int16_t));
        if (!c->coef){ j->err = "out of memory"; return 1; }
    }
    j->frame = 1;
    return 0;
}

unsigned char *jpeg_decode(const unsigned char *data, size_t size, int *w, int *h, int *channels, int want){
    Jpeg jd, *j = &jd;
    memset(j, 0, sizeof(*j));
    j->data = data; j->end = data + size;
    if (size < 4 || data[0] != 0xFF || data[1] != 0xD8) return fail(j, "not a JPEG file");
    const uint8_t *p = data + 2;
    int done = 0;
    while (!done){
        while (p < j->end && *p != 0xFF) p++;               // (junk between segments)
        while (p < j->end && *p == 0xFF) p++;
        if (p >= j->end) break;                              // no EOI: what was decoded stands
        int m = *p++;
        if (m == 0xD9){ done = 1; break; }
        if (m == 0xD8 || (m >= 0xD0 && m <= 0xD7) || m == 0x01) continue;   // no length
        if (p + 2 > j->end) return fail(j, "truncated");
        int len = be16(p);
        if (len < 2 || p + len > j->end) return fail(j, "truncated segment");
        const uint8_t *s = p + 2;
        int n = len - 2;
        switch (m){
        case 0xC0: case 0xC1: case 0xC2:
            if (frame(j, s, n, m)) return fail(j, j->err);
            break;
        case 0xC3: case 0xC5: case 0xC6: case 0xC7: case 0xC9: case 0xCA: case 0xCB: case 0xCD: case 0xCE: case 0xCF:
            return fail(j, "lossless, hierarchical or arithmetic-coded JPEG is not supported");
        case 0xC4:                                           // DHT: one or more tables
            while (n > 0){
                if (n < 17) return fail(j, "bad Huffman table");
                int tc = s[0] >> 4, th = s[0] & 15, total = 0;
                for (int i = 0; i < 16; i++) total += s[1 + i];
                if (tc > 1 || th > 3 || total > 256 || n < 17 + total) return fail(j, "bad Huffman table");
                if (huff_build(tc ? &j->ac[th] : &j->dc[th], s + 1, s + 17, total)) return fail(j, "bad Huffman table");
                s += 17 + total; n -= 17 + total;
            }
            break;
        case 0xDB:                                           // DQT
            while (n > 0){
                int pq = s[0] >> 4, tq = s[0] & 15, sz = pq ? 128 : 64;
                if (pq > 1 || tq > 3 || n < 1 + sz) return fail(j, "bad quantization table");
                for (int k = 0; k < 64; k++) j->q[tq][ZZ[k]] = (uint16_t)(pq ? be16(s + 1 + 2 * k) : s[1 + k]);
                s += 1 + sz; n -= 1 + sz;
            }
            break;
        case 0xDD:
            if (n < 2) return fail(j, "bad restart interval");
            j->restart = be16(s);
            break;
        case 0xDA: {
            if (!j->frame) return fail(j, "a scan before the frame");
            const uint8_t *nx = scan(j, s, len);
            if (!nx) return fail(j, j->err ? j->err : "bad scan");
            p = nx;
            continue;
        }
        case 0xEE:                                           // APP14: Adobe's colour transform
            if (n >= 12 && !memcmp(s, "Adobe", 5)){ j->adobe = 1; j->transform = s[11]; }
            break;
        default:
            break;                                           // APPn, COM, ...: skipped
        }
        p += len;
    }
    if (!j->frame) return fail(j, "no image");
    // the blocks to samples
    for (int i = 0; i < j->ncomp; i++){
        Comp *c = &j->comp[i];
        size_t pw = (size_t)c->bw * 8;
        c->pix = malloc(pw * c->bh * 8);
        if (!c->pix) return fail(j, "out of memory");
        const uint16_t *q = j->q[c->tq];
        for (int by = 0; by < c->bh; by++)
            for (int bx = 0; bx < c->bw; bx++)
                idct_block(c->coef + ((size_t)by * c->bw + bx) * 64, q, c->pix + (size_t)by * 8 * pw + (size_t)bx * 8, (int)pw);
        free(c->coef);
        c->coef = NULL;
    }
    int W = j->w, H = j->h;
    uint8_t *plane[3] = {NULL, NULL, NULL};
    for (int i = 0; i < j->ncomp; i++){
        plane[i] = malloc((size_t)W * H);
        if (!plane[i]){ for (int k = 0; k < i; k++) free(plane[k]); return fail(j, "out of memory"); }
        upsample(&j->comp[i], j->hmax, j->vmax, W, H, plane[i]);
    }
    int own = j->ncomp == 1 ? 1 : 3;
    int nc = want >= 1 && want <= 4 ? want : own;
    unsigned char *out = malloc((size_t)W * H * nc);
    if (!out){ for (int k = 0; k < j->ncomp; k++) free(plane[k]); return fail(j, "out of memory"); }
    // RGB: Adobe's transform 0, or components named R, G, B; else YCbCr (JFIF)
    int rgb = j->ncomp == 3 && ((j->adobe && j->transform == 0) ||
              (j->comp[0].id == 'R' && j->comp[1].id == 'G' && j->comp[2].id == 'B'));
    for (size_t i = 0; i < (size_t)W * H; i++){
        uint8_t c[3];
        if (own == 1) c[0] = c[1] = c[2] = plane[0][i];
        else if (rgb){ c[0] = plane[0][i]; c[1] = plane[1][i]; c[2] = plane[2][i]; }
        else ycc_to_rgb(plane[0][i], plane[1][i], plane[2][i], c);
        unsigned char *o = out + i * nc;
        if (nc == 1) o[0] = own == 1 ? c[0] : (uint8_t)((c[0] * 77 + c[1] * 150 + c[2] * 29 + 128) >> 8);
        else if (nc == 2){ o[0] = own == 1 ? c[0] : (uint8_t)((c[0] * 77 + c[1] * 150 + c[2] * 29 + 128) >> 8); o[1] = 255; }
        else { o[0] = c[0]; o[1] = c[1]; o[2] = c[2]; if (nc == 4) o[3] = 255; }
    }
    for (int k = 0; k < j->ncomp; k++){ free(plane[k]); free(j->comp[k].pix); }
    *w = W; *h = H; *channels = nc;
    return out;
}

unsigned char *jpeg_read(const char *path, int *w, int *h, int *channels, int want){
    FILE *f = fopen(path, "rb");
    if (!f){ fprintf(stderr, "jpeg: cannot open %s\n", path); return NULL; }
    size_t cap = 1 << 16, n = 0, k;
    unsigned char *buf = malloc(cap);
    while (buf && (k = fread(buf + n, 1, cap - n, f)) > 0){
        n += k;
        if (n == cap){
            unsigned char *q = realloc(buf, cap * 2);
            if (!q){ free(buf); buf = NULL; break; }
            buf = q; cap *= 2;
        }
    }
    fclose(f);
    if (!buf){ fprintf(stderr, "jpeg: out of memory\n"); return NULL; }
    unsigned char *px = jpeg_decode(buf, n, w, h, channels, want);
    free(buf);
    return px;
}
