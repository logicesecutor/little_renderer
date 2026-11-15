// front.c - WGSL tokens and the syntax tree (wgsl_impl.h)
#include "gpu/wgsl/wgsl_impl.h"

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void *wgsl_alloc(Wgsl *w, size_t n){
    void *p = calloc(1, n ? n : 1);
    void **b = realloc(w->blocks, (size_t)(w->nblocks + 1) * sizeof(*b));
    if (!p || !b){ free(p); if (b) w->blocks = b; wgsl_fail(w, 0, "out of memory"); }
    w->blocks = b;
    w->blocks[w->nblocks++] = p;
    return p;
}

void wgsl_fail(Wgsl *w, int line, const char *fmt, ...){
    char msg[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    // (the engine's WGSL arrives as one line: the C preprocessor's stringification; so a
    // position is given as the text around it)
    if (line > 0 && w->src){
        size_t at = (size_t)line - 1;
        if (at > w->len) at = w->len;
        size_t lo = at > 40 ? at - 40 : 0, hi = at + 40 < w->len ? at + 40 : w->len;
        snprintf(w->err, w->errlen, "%s, near \"%.*s\"", msg, (int)(hi - lo), w->src + lo);
    } else snprintf(w->err, w->errlen, "%s", msg);
    longjmp(w->fail, 1);
}

int wgsl_name_is(const char *s, int n, const char *lit){ return (int)strlen(lit) == n && !memcmp(s, lit, (size_t)n); }

// ---- tokens ----
static void push(Wgsl *w, Tok t){
    if (w->ntoks == w->cap_toks){
        int cap = w->cap_toks ? 2 * w->cap_toks : 1024;
        Tok *v = realloc(w->toks, (size_t)cap * sizeof(Tok));
        if (!v) wgsl_fail(w, 0, "out of memory");
        w->toks = v; w->cap_toks = cap;
    }
    w->toks[w->ntoks++] = t;
}

void wgsl_lex(Wgsl *w){
    const char *s = w->src, *e = w->src + w->len;
    while (s < e){
        if (isspace((unsigned char)*s)){ s++; continue; }
        if (s + 1 < e && s[0] == '/' && s[1] == '/'){ while (s < e && *s != '\n') s++; continue; }
        if (s + 1 < e && s[0] == '/' && s[1] == '*'){
            int depth = 0;
            do {
                if (s + 1 < e && s[0] == '/' && s[1] == '*'){ depth++; s += 2; }
                else if (s + 1 < e && s[0] == '*' && s[1] == '/'){ depth--; s += 2; }
                else s++;
            } while (s < e && depth > 0);
            continue;
        }
        Tok t;
        memset(&t, 0, sizeof(t));
        t.s = s;
        t.line = (int)(s - w->src) + 1;       // (a position, see wgsl_fail)
        if (isalpha((unsigned char)*s) || *s == '_'){
            while (s < e && (isalnum((unsigned char)*s) || *s == '_')) s++;
            t.k = TK_IDENT;
        } else if (isdigit((unsigned char)*s) || (*s == '.' && s + 1 < e && isdigit((unsigned char)s[1]))){
            const char *p = s;
            int is_float = 0;
            if (p[0] == '0' && p + 1 < e && (p[1] == 'x' || p[1] == 'X')){
                p += 2;
                while (p < e && isxdigit((unsigned char)*p)) p++;
                t.ival = strtoull(s + 2, NULL, 16);
            } else {
                while (p < e && isdigit((unsigned char)*p)) p++;
                if (p < e && *p == '.'){ is_float = 1; p++; while (p < e && isdigit((unsigned char)*p)) p++; }
                if (p < e && (*p == 'e' || *p == 'E')){
                    is_float = 1; p++;
                    if (p < e && (*p == '+' || *p == '-')) p++;
                    while (p < e && isdigit((unsigned char)*p)) p++;
                }
                char buf[64];
                int n = (int)(p - s) < 63 ? (int)(p - s) : 63;
                memcpy(buf, s, (size_t)n);
                buf[n] = 0;
                if (is_float) t.fval = strtod(buf, NULL);
                else t.ival = strtoull(buf, NULL, 10);
            }
            if (p < e && (*p == 'u' || *p == 'i')){ t.suffix = *p++; }
            else if (p < e && (*p == 'f' || *p == 'h')){ t.suffix = 'f'; is_float = 1; p++; }
            if (is_float && t.fval == 0 && !t.ival){
                // ("1f": an integer's digits with the f suffix)
                char buf[64];
                int n = (int)(p - s) < 63 ? (int)(p - s) : 63;
                memcpy(buf, s, (size_t)n);
                buf[n] = 0;
                t.fval = strtod(buf, NULL);
            }
            t.k = is_float ? TK_FLOAT : TK_INT;
            s = p;
        } else {
            static const struct { const char *s; TokKind k; } ops[] = {
                {"<<=", TK_SHLEQ}, {">>=", TK_SHREQ},
                {"->", TK_ARROW}, {"<=", TK_LE}, {">=", TK_GE}, {"==", TK_EQEQ}, {"!=", TK_NE}, {"&&", TK_ANDAND},
                {"||", TK_OROR}, {"<<", TK_SHL}, {">>", TK_SHR}, {"+=", TK_PLUSEQ}, {"-=", TK_MINUSEQ},
                {"*=", TK_STAREQ}, {"/=", TK_SLASHEQ}, {"%=", TK_PERCENTEQ}, {"&=", TK_AMPEQ}, {"|=", TK_PIPEEQ},
                {"^=", TK_CARETEQ}, {"++", TK_PLUSPLUS}, {"--", TK_MINUSMINUS},
                {"(", TK_LPAREN}, {")", TK_RPAREN}, {"{", TK_LBRACE}, {"}", TK_RBRACE}, {"[", TK_LBRACKET},
                {"]", TK_RBRACKET}, {",", TK_COMMA}, {";", TK_SEMI}, {":", TK_COLON}, {".", TK_DOT}, {"@", TK_AT},
                {"+", TK_PLUS}, {"-", TK_MINUS}, {"*", TK_STAR}, {"/", TK_SLASH}, {"%", TK_PERCENT}, {"&", TK_AMP},
                {"|", TK_PIPE}, {"^", TK_CARET}, {"~", TK_TILDE}, {"!", TK_BANG}, {"<", TK_LT}, {">", TK_GT}, {"=", TK_EQ}};
            int found = 0;
            for (size_t i = 0; i < sizeof(ops) / sizeof(ops[0]); i++){
                size_t n = strlen(ops[i].s);
                if ((size_t)(e - s) >= n && !memcmp(s, ops[i].s, n)){ t.k = ops[i].k; s += n; found = 1; break; }
            }
            if (!found) wgsl_fail(w, t.line, "unexpected character '%c'", *s);
        }
        t.len = (int)(s - t.s);
        push(w, t);
    }
    Tok eof;
    memset(&eof, 0, sizeof(eof));
    eof.k = TK_EOF;
    eof.line = (int)w->len;
    push(w, eof);
}

// ---- the parser ----
static Tok *peek(Wgsl *w){ return &w->toks[w->pos]; }
static Tok *peek2(Wgsl *w){ return &w->toks[w->pos + 1 < w->ntoks ? w->pos + 1 : w->pos]; }
static Tok *next(Wgsl *w){ Tok *t = &w->toks[w->pos]; if (t->k != TK_EOF) w->pos++; return t; }
static int is(Wgsl *w, TokKind k){ return peek(w)->k == k; }
static int is_word(Wgsl *w, const char *word){ Tok *t = peek(w); return t->k == TK_IDENT && wgsl_name_is(t->s, t->len, word); }
static int accept(Wgsl *w, TokKind k){ if (is(w, k)){ next(w); return 1; } return 0; }
static int accept_word(Wgsl *w, const char *word){ if (is_word(w, word)){ next(w); return 1; } return 0; }
static Tok *expect(Wgsl *w, TokKind k, const char *what){
    if (!is(w, k)) wgsl_fail(w, peek(w)->line, "expected %s", what);
    return next(w);
}
// a '>' closing a template list: also the first half of '>>', '>=', '>>='
static void expect_close(Wgsl *w){
    Tok *t = peek(w);
    if (t->k == TK_GT){ next(w); return; }
    if (t->k == TK_SHR){ t->k = TK_GT; t->s++; t->len--; return; }
    if (t->k == TK_GE){ t->k = TK_EQ; t->s++; t->len--; return; }
    if (t->k == TK_SHREQ){ t->k = TK_GE; t->s++; t->len--; return; }
    wgsl_fail(w, t->line, "expected '>'");
}

static Node *node(Wgsl *w, NodeKind k, int line){
    Node *n = wgsl_alloc(w, sizeof(Node));
    n->k = k;
    n->line = line;
    return n;
}

static void list_add(Wgsl *w, Node ***list, int *n, Node *x){
    Node **v = wgsl_alloc(w, (size_t)(*n + 1) * sizeof(Node *));
    if (*n) memcpy(v, *list, (size_t)*n * sizeof(Node *));
    v[(*n)++] = x;
    *list = v;
}

// the names that take template arguments in our WGSL: types, and bitcast
static int templated(const char *s, int n){
    static const char *names[] = {"vec2", "vec3", "vec4", "mat2x2", "mat2x3", "mat2x4", "mat3x2", "mat3x3", "mat3x4",
                                  "mat4x2", "mat4x3", "mat4x4", "array", "atomic", "ptr", "texture_2d", "bitcast"};
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) if (wgsl_name_is(s, n, names[i])) return 1;
    return 0;
}

static Node *expr(Wgsl *w);
static Node *binary(Wgsl *w, int min);
static Node *type(Wgsl *w);

// a type: a name, perhaps with template arguments (types, expressions, address spaces)
static Node *type(Wgsl *w){
    Tok *t = expect(w, TK_IDENT, "a type");
    Node *n = node(w, N_TYPE, t->line);
    n->name = t->s; n->nlen = t->len;
    if (templated(t->s, t->len) && accept(w, TK_LT)){
        do {
            Tok *a = peek(w);
            Node *arg;
            if (a->k == TK_IDENT && (wgsl_name_is(a->s, a->len, "function") || wgsl_name_is(a->s, a->len, "private") ||
                wgsl_name_is(a->s, a->len, "workgroup") || wgsl_name_is(a->s, a->len, "storage") ||
                wgsl_name_is(a->s, a->len, "uniform") || wgsl_name_is(a->s, a->len, "read") ||
                wgsl_name_is(a->s, a->len, "write") || wgsl_name_is(a->s, a->len, "read_write"))){
                next(w);
                arg = node(w, N_IDENT, a->line);
                arg->name = a->s; arg->nlen = a->len;
            } else if (a->k == TK_IDENT && (peek2(w)->k == TK_COMMA || peek2(w)->k == TK_GT || peek2(w)->k == TK_SHR ||
                                            peek2(w)->k == TK_GE || peek2(w)->k == TK_SHREQ ||
                                            (peek2(w)->k == TK_LT && templated(a->s, a->len)))){
                arg = type(w);                          // a type, or a constant's name (an array's length)
            } else arg = binary(w, 9);                  // an array's length (no comparisons or shifts: '>', '>>' end the list)
            list_add(w, &n->list, &n->n, arg);
        } while (accept(w, TK_COMMA));
        expect_close(w);
    }
    return n;
}

static Node *primary(Wgsl *w){
    Tok *t = next(w);
    Node *n;
    switch (t->k){
    case TK_INT:
        n = node(w, N_INT, t->line); n->ival = t->ival; n->suffix = t->suffix; return n;
    case TK_FLOAT:
        n = node(w, N_FLOAT, t->line); n->fval = t->fval; n->suffix = t->suffix; return n;
    case TK_LPAREN:
        n = expr(w);
        expect(w, TK_RPAREN, "')'");
        return n;
    case TK_IDENT:
        if (wgsl_name_is(t->s, t->len, "true") || wgsl_name_is(t->s, t->len, "false")){
            n = node(w, N_BOOL, t->line); n->ival = t->s[0] == 't'; return n;
        }
        if (templated(t->s, t->len) && is(w, TK_LT)){
            w->pos--;                                   // a templated constructor or bitcast
            Node *ty = type(w);
            n = node(w, N_CALL, t->line);
            n->name = ty->name; n->nlen = ty->nlen;
            n->type = ty;
        } else if (is(w, TK_LPAREN)){
            n = node(w, N_CALL, t->line);
            n->name = t->s; n->nlen = t->len;
        } else {
            n = node(w, N_IDENT, t->line); n->name = t->s; n->nlen = t->len;
            return n;
        }
        expect(w, TK_LPAREN, "'('");
        if (!is(w, TK_RPAREN)){
            do {
                if (is(w, TK_RPAREN)) break;            // (a trailing comma)
                list_add(w, &n->list, &n->n, expr(w));
            } while (accept(w, TK_COMMA));
        }
        expect(w, TK_RPAREN, "')'");
        return n;
    default:
        wgsl_fail(w, t->line, "expected an expression");
    }
    return NULL;
}

static Node *postfix(Wgsl *w){
    Node *n = primary(w);
    for (;;){
        Tok *t = peek(w);
        if (t->k == TK_LBRACKET){
            next(w);
            Node *x = node(w, N_INDEX, t->line);
            x->a = n; x->b = expr(w);
            expect(w, TK_RBRACKET, "']'");
            n = x;
        } else if (t->k == TK_DOT){
            next(w);
            Tok *m = expect(w, TK_IDENT, "a member");
            Node *x = node(w, N_MEMBER, m->line);
            x->a = n; x->name = m->s; x->nlen = m->len;
            n = x;
        } else return n;
    }
}

static Node *unary(Wgsl *w){
    Tok *t = peek(w);
    if (t->k == TK_MINUS || t->k == TK_BANG || t->k == TK_TILDE){
        next(w);
        Node *n = node(w, N_UNARY, t->line);
        n->op = t->k; n->a = unary(w);
        return n;
    }
    if (t->k == TK_AMP){ next(w); Node *n = node(w, N_ADDR, t->line); n->a = unary(w); return n; }
    if (t->k == TK_STAR){ next(w); Node *n = node(w, N_DEREF, t->line); n->a = unary(w); return n; }
    return postfix(w);
}

// precedence climbing (WGSL forbids mixing some operators without parentheses: we do not check)
static int prec(TokKind k){
    switch (k){
    case TK_OROR: return 1;
    case TK_ANDAND: return 2;
    case TK_PIPE: return 3;
    case TK_CARET: return 4;
    case TK_AMP: return 5;
    case TK_EQEQ: case TK_NE: return 6;
    case TK_LT: case TK_GT: case TK_LE: case TK_GE: return 7;
    case TK_SHL: case TK_SHR: return 8;
    case TK_PLUS: case TK_MINUS: return 9;
    case TK_STAR: case TK_SLASH: case TK_PERCENT: return 10;
    default: return 0;
    }
}

static Node *binary(Wgsl *w, int min){
    Node *l = unary(w);
    for (;;){
        Tok *t = peek(w);
        int p = prec(t->k);
        if (!p || p < min) return l;
        next(w);
        Node *n = node(w, N_BINARY, t->line);
        n->op = t->k; n->a = l; n->b = binary(w, p + 1);
        l = n;
    }
}

static Node *expr(Wgsl *w){ return binary(w, 1); }

static Node *block(Wgsl *w);

// an assignment, increment, call or declaration (the parts of a for's header too)
static Node *simple(Wgsl *w){
    Tok *t = peek(w);
    if (t->k == TK_IDENT && (wgsl_name_is(t->s, t->len, "let") || wgsl_name_is(t->s, t->len, "var") ||
                             wgsl_name_is(t->s, t->len, "const"))){
        next(w);
        Node *n = node(w, t->s[0] == 'l' ? N_LET : t->s[0] == 'v' ? N_VAR : N_CONST, t->line);
        if (n->k == N_VAR && accept(w, TK_LT)){              // var<function>
            expect(w, TK_IDENT, "an address space");
            expect_close(w);
        }
        Tok *name = expect(w, TK_IDENT, "a name");
        n->name = name->s; n->nlen = name->len;
        if (accept(w, TK_COLON)) n->type = type(w);
        if (accept(w, TK_EQ)) n->a = expr(w);
        else if (n->k != N_VAR) wgsl_fail(w, name->line, "'%.*s' needs a value", name->len, name->s);
        return n;
    }
    if (t->k == TK_IDENT && wgsl_name_is(t->s, t->len, "_") && peek2(w)->k == TK_EQ){
        next(w); next(w);                                     // _ = e: evaluated, dropped
        Node *n = node(w, N_EXPR, t->line);
        n->a = expr(w);
        return n;
    }
    Node *lhs = unary(w);
    Tok *op = peek(w);
    switch (op->k){
    case TK_EQ: case TK_PLUSEQ: case TK_MINUSEQ: case TK_STAREQ: case TK_SLASHEQ: case TK_PERCENTEQ:
    case TK_AMPEQ: case TK_PIPEEQ: case TK_CARETEQ: case TK_SHLEQ: case TK_SHREQ: {
        next(w);
        Node *n = node(w, N_ASSIGN, op->line);
        n->op = op->k; n->a = lhs; n->b = expr(w);
        return n;
    }
    case TK_PLUSPLUS: case TK_MINUSMINUS: {
        next(w);
        Node *n = node(w, N_INCDEC, op->line);
        n->op = op->k; n->a = lhs;
        return n;
    }
    default: {
        if (lhs->k != N_CALL) wgsl_fail(w, op->line, "expected an assignment or a call");
        Node *n = node(w, N_EXPR, lhs->line);
        n->a = lhs;
        return n;
    }
    }
}

static Node *statement(Wgsl *w){
    Tok *t = peek(w);
    if (t->k == TK_SEMI){ next(w); return NULL; }
    if (t->k == TK_LBRACE) return block(w);
    if (t->k == TK_IDENT){
        if (accept_word(w, "if")){
            Node *n = node(w, N_IF, t->line);
            n->a = expr(w);
            n->b = block(w);
            if (accept_word(w, "else")){
                if (is_word(w, "if")) n->c = statement(w);
                else n->c = block(w);
            }
            return n;
        }
        if (accept_word(w, "for")){
            Node *n = node(w, N_FOR, t->line);
            expect(w, TK_LPAREN, "'('");
            if (!is(w, TK_SEMI)) n->a = simple(w);
            expect(w, TK_SEMI, "';'");
            if (!is(w, TK_SEMI)) n->b = expr(w);
            expect(w, TK_SEMI, "';'");
            if (!is(w, TK_RPAREN)) n->c = simple(w);
            expect(w, TK_RPAREN, "')'");
            n->d = block(w);
            return n;
        }
        if (accept_word(w, "while")){
            Node *n = node(w, N_WHILE, t->line);
            n->a = expr(w);
            n->b = block(w);
            return n;
        }
        if (accept_word(w, "loop")){
            Node *n = node(w, N_LOOP, t->line);
            n->a = block(w);
            return n;
        }
        if (accept_word(w, "break")){
            Node *n;
            if (accept_word(w, "if")){ n = node(w, N_BREAKIF, t->line); n->a = expr(w); }
            else n = node(w, N_BREAK, t->line);
            expect(w, TK_SEMI, "';'");
            return n;
        }
        if (accept_word(w, "continue")){ expect(w, TK_SEMI, "';'"); return node(w, N_CONTINUE, t->line); }
        if (accept_word(w, "discard")){ expect(w, TK_SEMI, "';'"); return node(w, N_DISCARD, t->line); }
        if (accept_word(w, "return")){
            Node *n = node(w, N_RETURN, t->line);
            if (!is(w, TK_SEMI)) n->a = expr(w);
            expect(w, TK_SEMI, "';'");
            return n;
        }
        if (is_word(w, "switch") || is_word(w, "continuing")) wgsl_fail(w, t->line, "'%.*s' is not supported", t->len, t->s);
    }
    Node *n = simple(w);
    expect(w, TK_SEMI, "';'");
    return n;
}

static Node *block(Wgsl *w){
    Tok *t = expect(w, TK_LBRACE, "'{'");
    Node *b = node(w, N_BLOCK, t->line);
    while (!is(w, TK_RBRACE)){
        if (is(w, TK_EOF)) wgsl_fail(w, t->line, "a block without its '}'");
        Node *s = statement(w);
        if (s) list_add(w, &b->list, &b->n, s);
    }
    next(w);
    return b;
}

// ---- module scope ----
typedef struct { int group, binding, builtin, location, flat, stage; Node *wg[3]; } AllAttrs;

static int builtin_of(Wgsl *w, Tok *t){
    static const struct { const char *s; int b; } names[] = {
        {"global_invocation_id", BI_GLOBAL_ID}, {"local_invocation_id", BI_LOCAL_ID},
        {"local_invocation_index", BI_LOCAL_INDEX}, {"workgroup_id", BI_WORKGROUP_ID},
        {"num_workgroups", BI_NUM_WORKGROUPS}, {"vertex_index", BI_VERTEX_INDEX},
        {"instance_index", BI_INSTANCE_INDEX}, {"position", BI_POSITION}, {"front_facing", BI_FRONT_FACING},
        {"frag_depth", BI_FRAG_DEPTH}, {"subgroup_invocation_id", BI_SUBGROUP_ID}, {"subgroup_size", BI_SUBGROUP_SIZE}};
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) if (wgsl_name_is(t->s, t->len, names[i].s)) return names[i].b;
    wgsl_fail(w, t->line, "unknown built-in '%.*s'", t->len, t->s);
    return 0;
}

static int int_arg(Wgsl *w){
    Tok *t = expect(w, TK_INT, "a number");
    return (int)t->ival;
}

static AllAttrs attributes(Wgsl *w){
    AllAttrs a;
    memset(&a, 0, sizeof(a));
    a.group = a.binding = a.location = -1;
    while (accept(w, TK_AT)){
        Tok *t = expect(w, TK_IDENT, "an attribute");
        if (wgsl_name_is(t->s, t->len, "group")){ expect(w, TK_LPAREN, "'('"); a.group = int_arg(w); expect(w, TK_RPAREN, "')'"); }
        else if (wgsl_name_is(t->s, t->len, "binding")){ expect(w, TK_LPAREN, "'('"); a.binding = int_arg(w); expect(w, TK_RPAREN, "')'"); }
        else if (wgsl_name_is(t->s, t->len, "location")){ expect(w, TK_LPAREN, "'('"); a.location = int_arg(w); expect(w, TK_RPAREN, "')'"); }
        else if (wgsl_name_is(t->s, t->len, "builtin")){
            expect(w, TK_LPAREN, "'('");
            a.builtin = builtin_of(w, expect(w, TK_IDENT, "a built-in"));
            expect(w, TK_RPAREN, "')'");
        } else if (wgsl_name_is(t->s, t->len, "interpolate")){
            expect(w, TK_LPAREN, "'('");
            Tok *k = expect(w, TK_IDENT, "an interpolation");
            a.flat = wgsl_name_is(k->s, k->len, "flat");
            while (!is(w, TK_RPAREN) && !is(w, TK_EOF)) next(w);
            expect(w, TK_RPAREN, "')'");
        } else if (wgsl_name_is(t->s, t->len, "compute")) a.stage = ST_COMPUTE;
        else if (wgsl_name_is(t->s, t->len, "vertex")) a.stage = ST_VERTEX;
        else if (wgsl_name_is(t->s, t->len, "fragment")) a.stage = ST_FRAGMENT;
        else if (wgsl_name_is(t->s, t->len, "workgroup_size")){
            expect(w, TK_LPAREN, "'('");
            for (int i = 0; i < 3; i++){
                a.wg[i] = expr(w);
                if (!accept(w, TK_COMMA)) break;
                if (is(w, TK_RPAREN)) break;
            }
            expect(w, TK_RPAREN, "')'");
        } else if (wgsl_name_is(t->s, t->len, "align") || wgsl_name_is(t->s, t->len, "size") ||
                   wgsl_name_is(t->s, t->len, "id") || wgsl_name_is(t->s, t->len, "invariant")){
            wgsl_fail(w, t->line, "attribute @%.*s is not supported", t->len, t->s);
        } else if (wgsl_name_is(t->s, t->len, "must_use")){
        } else wgsl_fail(w, t->line, "unknown attribute @%.*s", t->len, t->s);
    }
    return a;
}

static Attrs field_attrs(AllAttrs a){
    Attrs f;
    memset(&f, 0, sizeof(f));
    f.builtin = a.builtin;
    f.location = a.location;
    f.flat = a.flat;
    return f;
}

static void add_global(Wgsl *w, GlobalDecl g){
    GlobalDecl *v = wgsl_alloc(w, (size_t)(w->nglobals + 1) * sizeof(*v));
    if (w->nglobals) memcpy(v, w->globals, (size_t)w->nglobals * sizeof(*v));
    v[w->nglobals++] = g;
    w->globals = v;
}

void wgsl_parse(Wgsl *w){
    w->pos = 0;
    while (!is(w, TK_EOF)){
        if (accept(w, TK_SEMI)) continue;
        AllAttrs at = attributes(w);
        Tok *t = expect(w, TK_IDENT, "a declaration");
        if (wgsl_name_is(t->s, t->len, "struct")){
            StructDecl s;
            memset(&s, 0, sizeof(s));
            Tok *name = expect(w, TK_IDENT, "a struct's name");
            s.name = name->s; s.nlen = name->len; s.line = name->line;
            expect(w, TK_LBRACE, "'{'");
            while (!is(w, TK_RBRACE)){
                AllAttrs fa = attributes(w);
                Tok *m = expect(w, TK_IDENT, "a member");
                expect(w, TK_COLON, "':'");
                Field f;
                memset(&f, 0, sizeof(f));
                f.name = m->s; f.nlen = m->len;
                f.type = type(w);
                f.at = field_attrs(fa);
                Field *v = wgsl_alloc(w, (size_t)(s.n + 1) * sizeof(Field));
                if (s.n) memcpy(v, s.f, (size_t)s.n * sizeof(Field));
                v[s.n++] = f;
                s.f = v;
                if (!accept(w, TK_COMMA)) break;
            }
            expect(w, TK_RBRACE, "'}'");
            accept(w, TK_SEMI);
            StructDecl *v = wgsl_alloc(w, (size_t)(w->nstructs + 1) * sizeof(*v));
            if (w->nstructs) memcpy(v, w->structs, (size_t)w->nstructs * sizeof(*v));
            v[w->nstructs++] = s;
            w->structs = v;
        } else if (wgsl_name_is(t->s, t->len, "var") || wgsl_name_is(t->s, t->len, "const")){
            GlobalDecl g;
            memset(&g, 0, sizeof(g));
            g.group = at.group; g.binding = at.binding;
            g.is_const = t->s[0] == 'c';
            g.space = g.is_const ? SP_NONE : SP_PRIVATE;
            if (!g.is_const && accept(w, TK_LT)){
                Tok *sp = expect(w, TK_IDENT, "an address space");
                g.space = wgsl_name_is(sp->s, sp->len, "private") ? SP_PRIVATE : wgsl_name_is(sp->s, sp->len, "workgroup") ? SP_WORKGROUP :
                          wgsl_name_is(sp->s, sp->len, "uniform") ? SP_UNIFORM : wgsl_name_is(sp->s, sp->len, "storage") ? SP_STORAGE : -1;
                if (g.space < 0) wgsl_fail(w, sp->line, "address space '%.*s' is not supported here", sp->len, sp->s);
                g.access = g.space == SP_STORAGE ? AC_READ : AC_NONE;
                if (accept(w, TK_COMMA)){
                    Tok *ac = expect(w, TK_IDENT, "an access mode");
                    g.access = wgsl_name_is(ac->s, ac->len, "read") ? AC_READ : wgsl_name_is(ac->s, ac->len, "read_write") ? AC_READ_WRITE : AC_WRITE;
                }
                expect_close(w);
            }
            Tok *name = expect(w, TK_IDENT, "a name");
            g.name = name->s; g.nlen = name->len; g.line = name->line;
            if (accept(w, TK_COLON)) g.type = type(w);
            if (accept(w, TK_EQ)) g.init = expr(w);
            expect(w, TK_SEMI, "';'");
            if (g.type && g.type->k == N_TYPE && !g.is_const && (wgsl_name_is(g.type->name, g.type->nlen, "texture_2d") ||
                wgsl_name_is(g.type->name, g.type->nlen, "texture_depth_2d") ||
                wgsl_name_is(g.type->name, g.type->nlen, "acceleration_structure"))) g.space = SP_HANDLE;
            if (g.is_const && !g.init) wgsl_fail(w, name->line, "a const needs a value");
            add_global(w, g);
        } else if (wgsl_name_is(t->s, t->len, "fn")){
            FnDecl f;
            memset(&f, 0, sizeof(f));
            Tok *name = expect(w, TK_IDENT, "a function's name");
            f.name = name->s; f.nlen = name->len; f.line = name->line;
            f.stage = at.stage;
            memcpy(f.wg, at.wg, sizeof(f.wg));
            expect(w, TK_LPAREN, "'('");
            while (!is(w, TK_RPAREN)){
                AllAttrs pa = attributes(w);
                Tok *p = expect(w, TK_IDENT, "a parameter");
                expect(w, TK_COLON, "':'");
                Field fp;
                memset(&fp, 0, sizeof(fp));
                fp.name = p->s; fp.nlen = p->len;
                fp.type = type(w);
                fp.at = field_attrs(pa);
                Field *v = wgsl_alloc(w, (size_t)(f.np + 1) * sizeof(Field));
                if (f.np) memcpy(v, f.p, (size_t)f.np * sizeof(Field));
                v[f.np++] = fp;
                f.p = v;
                if (!accept(w, TK_COMMA)) break;
            }
            expect(w, TK_RPAREN, "')'");
            if (accept(w, TK_ARROW)){
                AllAttrs ra = attributes(w);
                f.ret_at = field_attrs(ra);
                f.ret = type(w);
            }
            f.body = block(w);
            FnDecl *v = wgsl_alloc(w, (size_t)(w->nfns + 1) * sizeof(*v));
            if (w->nfns) memcpy(v, w->fns, (size_t)w->nfns * sizeof(*v));
            v[w->nfns++] = f;
            w->fns = v;
        } else if (wgsl_name_is(t->s, t->len, "enable") || wgsl_name_is(t->s, t->len, "requires") ||
                   wgsl_name_is(t->s, t->len, "diagnostic")){
            while (!is(w, TK_SEMI) && !is(w, TK_EOF)) next(w);
            accept(w, TK_SEMI);
        } else if (wgsl_name_is(t->s, t->len, "alias") || wgsl_name_is(t->s, t->len, "override") ||
                   wgsl_name_is(t->s, t->len, "const_assert")){
            wgsl_fail(w, t->line, "'%.*s' is not supported", t->len, t->s);
        } else wgsl_fail(w, t->line, "unexpected '%.*s' at module scope", t->len, t->s);
    }
}
