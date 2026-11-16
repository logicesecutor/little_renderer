// json.c - see json.h. A recursive descent parser over the text.
#include "formats/json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

typedef struct {
    const char *p, *end, *start;
    int depth, bad;
} Parser;

static void error(Parser *ps, const char *what){
    if (ps->bad) return;
    int line = 1;
    for (const char *q = ps->start; q < ps->p && q < ps->end; q++) line += *q == '\n';
    fprintf(stderr, "json: %s (line %d)\n", what, line);
    ps->bad = 1;
}

static void skip_ws(Parser *ps){
    while (ps->p < ps->end && (*ps->p == ' ' || *ps->p == '\t' || *ps->p == '\n' || *ps->p == '\r')) ps->p++;
}

static int hex4(Parser *ps, unsigned *out){
    if (ps->end - ps->p < 4) return 1;
    unsigned v = 0;
    for (int i = 0; i < 4; i++){
        char c = ps->p[i];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= (unsigned)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (unsigned)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (unsigned)(c - 'A' + 10);
        else return 1;
    }
    ps->p += 4;
    *out = v;
    return 0;
}

static char *parse_string(Parser *ps){
    ps->p++;                                           // the opening quote
    // the decoded string is never longer than the source
    const char *q = ps->p;
    while (q < ps->end && *q != '"'){ if (*q == '\\') q++; q++; }
    if (q >= ps->end){ error(ps, "unterminated string"); return NULL; }
    char *out = malloc((size_t)(q - ps->p) + 1), *o = out;
    if (!out){ error(ps, "out of memory"); return NULL; }
    while (ps->p < ps->end && *ps->p != '"'){
        unsigned char c = (unsigned char)*ps->p++;
        if (c < 0x20){ error(ps, "control character in a string"); free(out); return NULL; }
        if (c != '\\'){ *o++ = (char)c; continue; }
        char e = *ps->p++;
        switch (e){
        case '"': *o++ = '"'; break;
        case '\\': *o++ = '\\'; break;
        case '/': *o++ = '/'; break;
        case 'b': *o++ = '\b'; break;
        case 'f': *o++ = '\f'; break;
        case 'n': *o++ = '\n'; break;
        case 'r': *o++ = '\r'; break;
        case 't': *o++ = '\t'; break;
        case 'u': {
            unsigned cp;
            if (hex4(ps, &cp)){ error(ps, "bad \\u escape"); free(out); return NULL; }
            if (cp >= 0xD800 && cp < 0xDC00 && ps->end - ps->p >= 6 && ps->p[0] == '\\' && ps->p[1] == 'u'){
                unsigned lo;
                ps->p += 2;
                if (hex4(ps, &lo) || lo < 0xDC00 || lo > 0xDFFF){ error(ps, "bad surrogate pair"); free(out); return NULL; }
                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            }
            // (6 escaped bytes always cover the up to 4 bytes of UTF-8 they become)
            if (cp < 0x80) *o++ = (char)cp;
            else if (cp < 0x800){ *o++ = (char)(0xC0 | cp >> 6); *o++ = (char)(0x80 | (cp & 63)); }
            else if (cp < 0x10000){ *o++ = (char)(0xE0 | cp >> 12); *o++ = (char)(0x80 | (cp >> 6 & 63)); *o++ = (char)(0x80 | (cp & 63)); }
            else { *o++ = (char)(0xF0 | cp >> 18); *o++ = (char)(0x80 | (cp >> 12 & 63)); *o++ = (char)(0x80 | (cp >> 6 & 63)); *o++ = (char)(0x80 | (cp & 63)); }
            break;
        }
        default: error(ps, "bad escape"); free(out); return NULL;
        }
    }
    ps->p++;                                           // the closing quote
    *o = 0;
    return out;
}

static int parse_value(Parser *ps, Json *v);
static void release(Json *j);

// grows a list of items (and keys) by one
static int push(Parser *ps, Json *v, int *cap){
    if (v->n == *cap){
        int nc = *cap ? *cap * 2 : 4;
        Json *ni = realloc(v->items, (size_t)nc * sizeof(Json));
        if (!ni){ error(ps, "out of memory"); return 1; }
        v->items = ni;
        if (v->type == JSON_OBJECT){
            char **nk = realloc(v->keys, (size_t)nc * sizeof(char *));
            if (!nk){ error(ps, "out of memory"); return 1; }
            v->keys = nk;
        }
        *cap = nc;
    }
    memset(&v->items[v->n], 0, sizeof(Json));
    return 0;
}

static int parse_value(Parser *ps, Json *v){
    memset(v, 0, sizeof(*v));
    skip_ws(ps);
    if (ps->p >= ps->end){ error(ps, "unexpected end"); return 1; }
    if (++ps->depth > 512){ error(ps, "nested too deeply"); return 1; }
    char c = *ps->p;
    int r = 0;
    if (c == '{' || c == '['){
        int obj = c == '{', cap = 0;
        char close = obj ? '}' : ']';
        v->type = obj ? JSON_OBJECT : JSON_ARRAY;
        ps->p++;
        skip_ws(ps);
        if (ps->p < ps->end && *ps->p == close) ps->p++;
        else for (;;){
            if (push(ps, v, &cap)){ r = 1; break; }
            if (obj){
                skip_ws(ps);
                if (ps->p >= ps->end || *ps->p != '"'){ error(ps, "expected a key"); r = 1; break; }
                char *k = parse_string(ps);
                if (!k){ r = 1; break; }
                v->keys[v->n] = k;
                skip_ws(ps);
                if (ps->p >= ps->end || *ps->p != ':'){ free(k); error(ps, "expected ':'"); r = 1; break; }
                ps->p++;
            }
            if (parse_value(ps, &v->items[v->n])){
                if (obj) free(v->keys[v->n]);
                release(&v->items[v->n]);
                r = 1;
                break;
            }
            v->n++;
            skip_ws(ps);
            if (ps->p < ps->end && *ps->p == ','){ ps->p++; continue; }
            if (ps->p < ps->end && *ps->p == close){ ps->p++; break; }
            error(ps, obj ? "expected ',' or '}'" : "expected ',' or ']'");
            r = 1;
            break;
        }
    } else if (c == '"'){
        v->type = JSON_STRING;
        v->str = parse_string(ps);
        r = !v->str;
    } else if (c == '-' || (c >= '0' && c <= '9')){
        char *e;
        v->type = JSON_NUMBER;
        v->num = strtod(ps->p, &e);        // (the text is NUL-terminated by json_parse)
        if (e == ps->p){ error(ps, "bad number"); r = 1; }
        ps->p = e;
    } else if (ps->end - ps->p >= 4 && !strncmp(ps->p, "true", 4)){ v->type = JSON_TRUE; ps->p += 4; }
    else if (ps->end - ps->p >= 5 && !strncmp(ps->p, "false", 5)){ v->type = JSON_FALSE; ps->p += 5; }
    else if (ps->end - ps->p >= 4 && !strncmp(ps->p, "null", 4)){ v->type = JSON_NULL; ps->p += 4; }
    else { error(ps, "unexpected character"); r = 1; }
    ps->depth--;
    return r;
}

// frees what a value holds (its strings and items), not the value itself
static void release(Json *j){
    if (j->type == JSON_STRING) free(j->str);
    for (int i = 0; i < j->n; i++){
        release(&j->items[i]);
        if (j->keys) free(j->keys[i]);
    }
    free(j->items);
    free(j->keys);
    memset(j, 0, sizeof(*j));
}

Json *json_parse(const char *text, size_t len){
    // a NUL-terminated copy (strtod reads up to a character that is not a number's)
    char *t = malloc(len + 1);
    Json *root = calloc(1, sizeof(Json));
    if (!t || !root){ free(t); free(root); fprintf(stderr, "json: out of memory\n"); return NULL; }
    memcpy(t, text, len);
    t[len] = 0;
    Parser ps = {t, t + len, t, 0, 0};
    if (len >= 3 && (unsigned char)t[0] == 0xEF && (unsigned char)t[1] == 0xBB && (unsigned char)t[2] == 0xBF) ps.p += 3;  // a BOM
    int r = parse_value(&ps, root);
    if (!r){
        skip_ws(&ps);
        if (ps.p < ps.end){ error(&ps, "text after the value"); r = 1; }
    }
    free(t);
    if (r){ release(root); free(root); return NULL; }
    return root;
}

void json_free(Json *j){
    if (!j) return;
    release(j);
    free(j);
}

const Json *json_get(const Json *obj, const char *key){
    if (!obj || obj->type != JSON_OBJECT) return NULL;
    for (int i = 0; i < obj->n; i++) if (!strcmp(obj->keys[i], key)) return &obj->items[i];
    return NULL;
}
const Json *json_at(const Json *arr, int i){
    return arr && arr->type == JSON_ARRAY && i >= 0 && i < arr->n ? &arr->items[i] : NULL;
}
int json_len(const Json *arr){ return arr && arr->type == JSON_ARRAY ? arr->n : 0; }
double json_num(const Json *v, double def){ return v && v->type == JSON_NUMBER ? v->num : def; }
int json_int(const Json *v, int def){
    // (outside int's range, or NaN: the default; converting it would be undefined)
    return v && v->type == JSON_NUMBER && v->num >= (double)INT_MIN && v->num <= (double)INT_MAX ? (int)v->num : def;
}
const char *json_str(const Json *v, const char *def){ return v && v->type == JSON_STRING ? v->str : def; }
