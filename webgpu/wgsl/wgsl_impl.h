// wgsl_impl.h - our WGSL compiler's insides: tokens, the syntax tree, types (wgsl.h).
//
// The WGSL we write (the engine's kernels), not the whole language: structs, module-scope
// var / const, functions with pointer parameters, let / var / const, if, for, while, loop,
// break, continue, return; the operators; constructors; the built-in functions the kernels
// call; atomics; workgroup memory and barriers; textureLoad on 2-D textures; compute,
// vertex and fragment entry points. Unsupported things are reported, never guessed.
#ifndef LR_WGSL_IMPL_H
#define LR_WGSL_IMPL_H

#include <stddef.h>
#include <stdint.h>
#include <setjmp.h>

// ---- tokens ----
typedef enum {
    TK_EOF, TK_IDENT, TK_INT, TK_FLOAT,
    TK_LPAREN, TK_RPAREN, TK_LBRACE, TK_RBRACE, TK_LBRACKET, TK_RBRACKET,
    TK_COMMA, TK_SEMI, TK_COLON, TK_DOT, TK_AT, TK_ARROW,
    TK_PLUS, TK_MINUS, TK_STAR, TK_SLASH, TK_PERCENT, TK_AMP, TK_PIPE, TK_CARET, TK_TILDE, TK_BANG,
    TK_LT, TK_GT, TK_LE, TK_GE, TK_EQEQ, TK_NE, TK_ANDAND, TK_OROR, TK_SHL, TK_SHR,
    TK_EQ, TK_PLUSEQ, TK_MINUSEQ, TK_STAREQ, TK_SLASHEQ, TK_PERCENTEQ, TK_AMPEQ, TK_PIPEEQ, TK_CARETEQ,
    TK_SHLEQ, TK_SHREQ, TK_PLUSPLUS, TK_MINUSMINUS
} TokKind;

typedef struct {
    TokKind k;
    const char *s; int len;          // the text
    int line;
    uint64_t ival; double fval;
    char suffix;                     // 'u', 'i', 'f', 0 (abstract)
} Tok;

// ---- the syntax tree ----
typedef enum {
    // expressions
    N_INT, N_FLOAT, N_BOOL, N_IDENT, N_UNARY, N_BINARY, N_CALL, N_INDEX, N_MEMBER, N_ADDR, N_DEREF,
    // statements
    N_BLOCK, N_LET, N_VAR, N_CONST, N_ASSIGN, N_INCDEC, N_IF, N_FOR, N_WHILE, N_LOOP, N_BREAK,
    N_BREAKIF, N_CONTINUE, N_RETURN, N_EXPR, N_DISCARD,
    // a type (in an expression: the callee of a constructor, a template argument)
    N_TYPE
} NodeKind;

typedef struct Node Node;
struct Node {
    NodeKind k;
    int line;
    int op;                          // the operator's token kind (unary, binary, assignment)
    const char *name; int nlen;      // identifiers, callees, members, type names
    uint64_t ival; double fval; char suffix;
    Node *a, *b, *c, *d;             // children: operands, condition, bodies, ...
    Node **list; int n;              // arguments, statements, template arguments
    Node *type;                      // a declared type (let, var, const, parameters)
    void *sema;                      // the checker's notes (a resolved type)
};

typedef struct {
    const char *name; int nlen;
    int builtin;                     // @builtin(...): a BI_*; 0 none
    int location;                    // @location(n); -1 none
    int flat;                        // @interpolate(flat)
} Attrs;

typedef struct { const char *name; int nlen; Node *type; Attrs at; } Field;

typedef struct {
    const char *name; int nlen;
    Field *f; int n;
    int line;
    void *sema;
} StructDecl;

typedef struct {
    const char *name; int nlen;
    int line;
    int space;                       // SP_*
    int access;                      // AC_*
    Node *type, *init;
    int group, binding;              // -1 none
    int is_const;                    // a module-scope const
    void *sema;
} GlobalDecl;

typedef struct {
    const char *name; int nlen;
    int line;
    Field *p; int np;                // parameters
    Node *ret; Attrs ret_at;
    int stage;                       // 0, ST_COMPUTE, ST_VERTEX, ST_FRAGMENT
    Node *wg[3];                     // @workgroup_size
    Node *body;
    void *sema;
} FnDecl;

enum { SP_NONE, SP_FUNCTION, SP_PRIVATE, SP_WORKGROUP, SP_UNIFORM, SP_STORAGE, SP_HANDLE };
enum { AC_NONE, AC_READ, AC_WRITE, AC_READ_WRITE };
enum { ST_NONE, ST_COMPUTE, ST_VERTEX, ST_FRAGMENT };
enum { BI_NONE, BI_GLOBAL_ID, BI_LOCAL_ID, BI_LOCAL_INDEX, BI_WORKGROUP_ID, BI_NUM_WORKGROUPS,
       BI_VERTEX_INDEX, BI_INSTANCE_INDEX, BI_POSITION, BI_FRONT_FACING, BI_FRAG_DEPTH,
       BI_SUBGROUP_ID, BI_SUBGROUP_SIZE };

typedef struct {
    // the source and where we are
    const char *src; size_t len;
    Tok *toks; int ntoks, cap_toks, pos;
    // the module
    StructDecl *structs; int nstructs;
    GlobalDecl *globals; int nglobals;
    FnDecl *fns; int nfns;
    // memory: everything comes from one arena, freed at the end
    void **blocks; int nblocks;
    // ... and what grows (the emitter's word buffers and tables: wgsl_grow), freed there too if
    // an error left before its owner did
    void **heap; int nheap, cap_heap;
    // errors: the first one ends the compilation
    char *err; size_t errlen;
    jmp_buf fail;
} Wgsl;

void *wgsl_alloc(Wgsl *w, size_t n);                    // zeroed, freed with the compilation
// Memory that grows, as realloc (old NULL: new), known to the compilation: what is still held
// when it ends is freed (an error leaves by longjmp, from anywhere). Never NULL: fails
// instead. wgsl_drop lets one go: freed, or (keep) left to the caller, the compilation's no more.
void *wgsl_grow(Wgsl *w, void *old, size_t n);
void wgsl_drop(Wgsl *w, void *p, int keep);
void wgsl_fail(Wgsl *w, int line, const char *fmt, ...);  // reports and leaves (longjmp)
void wgsl_lex(Wgsl *w);
void wgsl_parse(Wgsl *w);
int wgsl_name_is(const char *s, int n, const char *lit);
// the SPIR-V of the parsed module (spirv.c)
void wgsl_emit(Wgsl *w, uint32_t **words, size_t *nwords);

#endif
