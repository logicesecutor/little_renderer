// spirv.c - the checked WGSL module as SPIR-V 1.4 for Vulkan (wgsl_impl.h): types and their
// layout, constants, module-scope variables, expressions, statements, entry points. One pass:
// each expression's type is worked out as it is emitted.
#include "gpu/wgsl/wgsl_impl.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---- SPIR-V numbers ----
enum {
    OpName = 5, OpMemberName = 6, OpExtInstImport = 11, OpExtInst = 12, OpMemoryModel = 14, OpEntryPoint = 15,
    OpExecutionMode = 16, OpCapability = 17, OpTypeVoid = 19, OpTypeBool = 20, OpTypeInt = 21, OpTypeFloat = 22,
    OpTypeVector = 23, OpTypeMatrix = 24, OpTypeImage = 25, OpTypeArray = 28, OpTypeRuntimeArray = 29,
    OpTypeStruct = 30, OpTypePointer = 32, OpTypeFunction = 33, OpConstantTrue = 41, OpConstantFalse = 42,
    OpConstant = 43, OpConstantComposite = 44, OpConstantNull = 46, OpFunction = 54, OpFunctionParameter = 55,
    OpFunctionEnd = 56, OpFunctionCall = 57, OpVariable = 59, OpLoad = 61, OpStore = 62, OpAccessChain = 65,
    OpArrayLength = 68, OpDecorate = 71, OpMemberDecorate = 72, OpVectorExtractDynamic = 77,
    OpVectorShuffle = 79, OpCompositeConstruct = 80, OpCompositeExtract = 81, OpCompositeInsert = 82,
    OpTranspose = 84, OpImageFetch = 95, OpImageQuerySizeLod = 103, OpConvertFToU = 109, OpConvertFToS = 110,
    OpConvertSToF = 111, OpConvertUToF = 112, OpBitcast = 124, OpSNegate = 126, OpFNegate = 127, OpIAdd = 128,
    OpFAdd = 129, OpISub = 130, OpFSub = 131, OpIMul = 132, OpFMul = 133, OpUDiv = 134, OpSDiv = 135,
    OpFDiv = 136, OpUMod = 137, OpSRem = 138, OpFRem = 140, OpVectorTimesScalar = 142, OpMatrixTimesScalar = 143,
    OpVectorTimesMatrix = 144, OpMatrixTimesVector = 145, OpMatrixTimesMatrix = 146, OpDot = 148, OpAny = 154,
    OpAll = 155, OpLogicalEqual = 164, OpLogicalNotEqual = 165, OpLogicalOr = 166, OpLogicalAnd = 167,
    OpLogicalNot = 168, OpSelect = 169, OpIEqual = 170, OpINotEqual = 171, OpUGreaterThan = 172,
    OpSGreaterThan = 173, OpUGreaterThanEqual = 174, OpSGreaterThanEqual = 175, OpULessThan = 176,
    OpSLessThan = 177, OpULessThanEqual = 178, OpSLessThanEqual = 179, OpFOrdEqual = 180, OpFUnordNotEqual = 183,
    OpFOrdLessThan = 184, OpFOrdGreaterThan = 186, OpFOrdLessThanEqual = 188, OpFOrdGreaterThanEqual = 190,
    OpShiftRightLogical = 194, OpShiftRightArithmetic = 195, OpShiftLeftLogical = 196, OpBitwiseOr = 197,
    OpBitwiseXor = 198, OpBitwiseAnd = 199, OpNot = 200, OpBitReverse = 204, OpBitCount = 205,
    OpDPdx = 207, OpDPdy = 208, OpFwidth = 209, OpDPdxFine = 210, OpDPdyFine = 211, OpDPdxCoarse = 213,
    OpDPdyCoarse = 214, OpControlBarrier = 224, OpAtomicLoad = 227, OpAtomicStore = 228, OpAtomicExchange = 229,
    OpAtomicCompareExchange = 230, OpAtomicIAdd = 234, OpAtomicISub = 235, OpAtomicSMin = 236,
    OpAtomicUMin = 237, OpAtomicSMax = 238, OpAtomicUMax = 239, OpAtomicAnd = 240, OpAtomicOr = 241,
    OpAtomicXor = 242, OpCopyLogical = 400, OpPhi = 245, OpLoopMerge = 246, OpSelectionMerge = 247, OpLabel = 248, OpBranch = 249,
    OpBranchConditional = 250, OpKill = 252, OpReturn = 253, OpReturnValue = 254, OpUnreachable = 255
};
enum { SC_UniformConstant = 0, SC_Input = 1, SC_Uniform = 2, SC_Output = 3, SC_Workgroup = 4, SC_Private = 6,
       SC_Function = 7, SC_StorageBuffer = 12 };
enum { DecBlock = 2, DecColMajor = 5, DecArrayStride = 6, DecMatrixStride = 7, DecBuiltIn = 11, DecFlat = 14,
       DecNonWritable = 24, DecLocation = 30, DecBinding = 33, DecDescriptorSet = 34, DecOffset = 35 };
enum { G450_Round = 1, G450_Trunc = 3, G450_FAbs = 4, G450_SAbs = 5, G450_FSign = 6, G450_SSign = 7,
       G450_Floor = 8, G450_Ceil = 9, G450_Fract = 10, G450_Sin = 13, G450_Cos = 14, G450_Tan = 15,
       G450_Asin = 16, G450_Acos = 17, G450_Atan = 18, G450_Sinh = 19, G450_Cosh = 20, G450_Tanh = 21,
       G450_Atan2 = 25, G450_Pow = 26, G450_Exp = 27, G450_Log = 28, G450_Exp2 = 29, G450_Log2 = 30,
       G450_Sqrt = 31, G450_InverseSqrt = 32, G450_Determinant = 33, G450_FMin = 37, G450_UMin = 38,
       G450_SMin = 39, G450_FMax = 40, G450_UMax = 41, G450_SMax = 42, G450_FClamp = 43, G450_UClamp = 44,
       G450_SClamp = 45, G450_FMix = 46, G450_Step = 48, G450_SmoothStep = 49, G450_Fma = 50,
       G450_PackSnorm4x8 = 54, G450_PackUnorm4x8 = 55, G450_PackSnorm2x16 = 56, G450_PackUnorm2x16 = 57,
       G450_PackHalf2x16 = 58, G450_UnpackSnorm2x16 = 60, G450_UnpackUnorm2x16 = 61, G450_UnpackHalf2x16 = 62,
       G450_UnpackSnorm4x8 = 63, G450_UnpackUnorm4x8 = 64, G450_Length = 66, G450_Distance = 67,
       G450_Cross = 68, G450_Normalize = 69, G450_FaceForward = 70, G450_Reflect = 71, G450_Refract = 72,
       G450_FindILsb = 73, G450_FindSMsb = 74, G450_FindUMsb = 75 };

// ---- word buffers ----
typedef struct { uint32_t *w; size_t n, cap; } Words;

static void put(Wgsl *W, Words *b, uint32_t x){
    if (b->n == b->cap){
        size_t cap = b->cap ? 2 * b->cap : 256;
        uint32_t *v = realloc(b->w, cap * sizeof(uint32_t));
        if (!v) wgsl_fail(W, 0, "out of memory");
        b->w = v; b->cap = cap;
    }
    b->w[b->n++] = x;
}

// ---- types ----
typedef enum { T_VOID, T_BOOL, T_I32, T_U32, T_F32, T_AINT, T_AFLOAT, T_VEC, T_MAT, T_ARRAY, T_RTARRAY,
               T_STRUCT, T_ATOMIC, T_PTR, T_TEX, T_TEXDEPTH } Kind;

typedef struct Type Type;
typedef struct StructInfo StructInfo;
struct Type {
    Kind k;
    Type *e;                         // the element / component / pointee (vectors: the scalar; matrices: the column vector)
    int n;                           // vector size, matrix columns, array length
    int sc;                          // pointers: the storage class
    StructInfo *s;
    uint32_t id;                     // its SPIR-V id (0: not yet)
    uint32_t lid;                    // the same with its memory layout decorated (buffers only: Vulkan
                                     // forbids the decorations in function, private, workgroup memory)
    Type *nextt;
};

struct StructInfo {
    const char *name; int nlen;
    int n;
    const char **mname; int *mlen;
    Type **mt;
    uint32_t *off;
    uint32_t size, align;
    Attrs *mat;                      // members' attributes (entry point interfaces)
    StructDecl *decl;
    int resolving;
};

typedef struct {
    Wgsl *W;
    Words caps, ext, mem, entries, modes, names, decos, globals, fns;
    Words fvars, fbody;              // the current function: its variables, its blocks
    uint32_t bound;
    uint32_t glsl;                   // the GLSL.std.450 import
    Type *types;
    int cap_image_query, cap_derivative;
    // constants already made: (type, bits) -> id; vectors: (type, components) -> id
    struct { Type *t; uint64_t bits; uint32_t id; } *consts; int nconsts, cap_consts;
    struct { Type *t; double v[16]; uint32_t id; } *cc; int ncc, cap_cc;
} Sp;

static uint32_t new_id(Sp *S){ return ++S->bound; }

static void op(Sp *S, Words *b, uint32_t opc, int n, const uint32_t *args){
    put(S->W, b, ((uint32_t)(n + 1) << 16) | opc);
    for (int i = 0; i < n; i++) put(S->W, b, args[i]);
}
#define OP(S, b, opc, ...) do { const uint32_t a_[] = {__VA_ARGS__}; op(S, b, opc, (int)(sizeof(a_) / sizeof(a_[0])), a_); } while (0)

// a string operand (nul terminated, padded to words)
static int str_words(int n){ return n / 4 + 1; }
static void op_str(Sp *S, Words *b, uint32_t opc, int nbefore, const uint32_t *before, const char *s, int slen,
                   int nafter, const uint32_t *after){
    int sw = str_words(slen);
    put(S->W, b, ((uint32_t)(1 + nbefore + sw + nafter) << 16) | opc);
    for (int i = 0; i < nbefore; i++) put(S->W, b, before[i]);
    for (int i = 0; i < sw; i++){
        uint32_t x = 0;
        for (int k = 0; k < 4; k++){
            int at = 4 * i + k;
            if (at < slen) x |= (uint32_t)(unsigned char)s[at] << (8 * k);
        }
        put(S->W, b, x);
    }
    for (int i = 0; i < nafter; i++) put(S->W, b, after[i]);
}

static Type *mk(Sp *S, Kind k, Type *e, int n, int sc, StructInfo *si){
    for (Type *t = S->types; t; t = t->nextt)
        if (t->k == k && t->e == e && t->n == n && t->sc == sc && t->s == si) return t;
    Type *t = wgsl_alloc(S->W, sizeof(Type));
    t->k = k; t->e = e; t->n = n; t->sc = sc; t->s = si;
    t->nextt = S->types;
    S->types = t;
    return t;
}
static Type *t_void(Sp *S){ return mk(S, T_VOID, NULL, 0, 0, NULL); }
static Type *t_bool(Sp *S){ return mk(S, T_BOOL, NULL, 0, 0, NULL); }
static Type *t_i32(Sp *S){ return mk(S, T_I32, NULL, 0, 0, NULL); }
static Type *t_u32(Sp *S){ return mk(S, T_U32, NULL, 0, 0, NULL); }
static Type *t_f32(Sp *S){ return mk(S, T_F32, NULL, 0, 0, NULL); }
static Type *t_aint(Sp *S){ return mk(S, T_AINT, NULL, 0, 0, NULL); }
static Type *t_afloat(Sp *S){ return mk(S, T_AFLOAT, NULL, 0, 0, NULL); }
static Type *t_vec(Sp *S, Type *e, int n){ return n == 1 ? e : mk(S, T_VEC, e, n, 0, NULL); }
static Type *t_mat(Sp *S, int cols, int rows){ return mk(S, T_MAT, t_vec(S, t_f32(S), rows), cols, 0, NULL); }
static Type *t_ptr(Sp *S, Type *e, int sc){ return mk(S, T_PTR, e, 0, sc, NULL); }

static int is_scalar(Type *t){ return t->k == T_BOOL || t->k == T_I32 || t->k == T_U32 || t->k == T_F32 || t->k == T_AINT || t->k == T_AFLOAT; }
static Type *scalar_of(Type *t){ return t->k == T_VEC ? t->e : t; }
static int width_of(Type *t){ return t->k == T_VEC ? t->n : 1; }
static int is_abstract(Type *t){ Type *s = scalar_of(t); return s->k == T_AINT || s->k == T_AFLOAT; }
static int is_float(Type *t){ Type *s = scalar_of(t); return s->k == T_F32 || s->k == T_AFLOAT; }
static int is_sint(Type *t){ Type *s = scalar_of(t); return s->k == T_I32 || s->k == T_AINT; }
static int is_uint(Type *t){ return scalar_of(t)->k == T_U32; }
static int is_int(Type *t){ return is_sint(t) || is_uint(t); }
static int is_bool(Type *t){ return scalar_of(t)->k == T_BOOL; }

static void type_name(Type *t, char *out, size_t n){
    switch (t->k){
    case T_VOID: snprintf(out, n, "void"); break;
    case T_BOOL: snprintf(out, n, "bool"); break;
    case T_I32: snprintf(out, n, "i32"); break;
    case T_U32: snprintf(out, n, "u32"); break;
    case T_F32: snprintf(out, n, "f32"); break;
    case T_AINT: snprintf(out, n, "abstract-int"); break;
    case T_AFLOAT: snprintf(out, n, "abstract-float"); break;
    case T_VEC: { char e[32]; type_name(t->e, e, sizeof(e)); snprintf(out, n, "vec%d<%s>", t->n, e); break; }
    case T_MAT: snprintf(out, n, "mat%dx%d<f32>", t->n, t->e->n); break;
    case T_ARRAY: case T_RTARRAY: { char e[48]; type_name(t->e, e, sizeof(e)); snprintf(out, n, t->k == T_ARRAY ? "array<%.40s, %d>" : "array<%.40s>", e, t->n); break; }
    case T_STRUCT: snprintf(out, n, "%.*s", t->s->nlen, t->s->name); break;
    case T_ATOMIC: snprintf(out, n, "atomic"); break;
    case T_PTR: { char e[48]; type_name(t->e, e, sizeof(e)); snprintf(out, n, "ptr<%.40s>", e); break; }
    case T_TEX: snprintf(out, n, "texture_2d<f32>"); break;
    case T_TEXDEPTH: snprintf(out, n, "texture_depth_2d"); break;
    }
}

// WGSL's memory layout (host-shareable): alignment and size
static uint32_t align_of(Type *t){
    switch (t->k){
    case T_VEC: return t->n == 2 ? 8 : 16;
    case T_MAT: return align_of(t->e);
    case T_ARRAY: case T_RTARRAY: return align_of(t->e);
    case T_STRUCT: return t->s->align;
    default: return 4;
    }
}
static uint32_t round_up(uint32_t a, uint32_t x){ return (x + a - 1) / a * a; }
static uint32_t size_of(Type *t){
    switch (t->k){
    case T_VEC: return 4u * (uint32_t)t->n;
    case T_MAT: return (uint32_t)t->n * round_up(align_of(t->e), size_of(t->e));
    case T_ARRAY: return (uint32_t)t->n * round_up(align_of(t->e), size_of(t->e));
    case T_RTARRAY: return round_up(align_of(t->e), size_of(t->e));
    case T_STRUCT: return t->s->size;
    default: return 4;
    }
}
static uint32_t stride_of(Type *t){ return round_up(align_of(t->e), size_of(t->e)); }

static uint32_t tid_l(Sp *S, Type *t);
static uint32_t tid(Sp *S, Type *t);
static int layout_differs(Type *t){ return t->k == T_ARRAY || t->k == T_RTARRAY || t->k == T_STRUCT; }

// the length of an array type: a u32 constant (made here: constants come after their types)
static uint32_t array_len(Sp *S, Type *t){
    uint32_t ut = 0;
    Type *u = mk(S, T_U32, NULL, 0, 0, NULL);
    if (!u->id){ u->id = new_id(S); OP(S, &S->globals, OpTypeInt, u->id, 32, 0); }
    ut = u->id;
    uint32_t len = new_id(S);
    OP(S, &S->globals, OpConstant, ut, len, (uint32_t)t->n);
    return len;
}

// a composite's SPIR-V type, plain (lay = 0) or with its memory layout (lay = 1)
static uint32_t composite(Sp *S, Type *t, int lay){
    uint32_t id;
    if (t->k == T_ARRAY){
        uint32_t e = lay ? tid_l(S, t->e) : tid(S, t->e), len = array_len(S, t);
        id = new_id(S);
        OP(S, &S->globals, OpTypeArray, id, e, len);
        if (lay) OP(S, &S->decos, OpDecorate, id, DecArrayStride, stride_of(t));
        return id;
    }
    if (t->k == T_RTARRAY){
        uint32_t e = tid_l(S, t->e);
        id = new_id(S);
        OP(S, &S->globals, OpTypeRuntimeArray, id, e);
        OP(S, &S->decos, OpDecorate, id, DecArrayStride, stride_of(t));
        return id;
    }
    StructInfo *si = t->s;
    uint32_t *m = malloc((size_t)(si->n ? si->n : 1) * sizeof(uint32_t));
    if (!m) wgsl_fail(S->W, 0, "out of memory");
    for (int i = 0; i < si->n; i++) m[i] = lay ? tid_l(S, si->mt[i]) : tid(S, si->mt[i]);
    id = new_id(S);
    put(S->W, &S->globals, ((uint32_t)(2 + si->n) << 16) | OpTypeStruct);
    put(S->W, &S->globals, id);
    for (int i = 0; i < si->n; i++) put(S->W, &S->globals, m[i]);
    free(m);
    op_str(S, &S->names, OpName, 1, &id, si->name, si->nlen, 0, NULL);
    for (int i = 0; i < si->n; i++){
        uint32_t pre[2] = {id, (uint32_t)i};
        op_str(S, &S->names, OpMemberName, 2, pre, si->mname[i], si->mlen[i], 0, NULL);
        if (!lay) continue;
        OP(S, &S->decos, OpMemberDecorate, id, (uint32_t)i, DecOffset, si->off[i]);
        Type *mt = si->mt[i];
        while (mt->k == T_ARRAY || mt->k == T_RTARRAY) mt = mt->e;
        if (mt->k == T_MAT){
            OP(S, &S->decos, OpMemberDecorate, id, (uint32_t)i, DecColMajor);
            OP(S, &S->decos, OpMemberDecorate, id, (uint32_t)i, DecMatrixStride, round_up(align_of(mt->e), size_of(mt->e)));
        }
    }
    return id;
}

static uint32_t tid_l(Sp *S, Type *t){
    if (!layout_differs(t)) return tid(S, t);
    if (!t->lid) t->lid = composite(S, t, 1);
    return t->lid;
}

static int buffer_sc(int sc){ return sc == SC_Uniform || sc == SC_StorageBuffer; }

// a type's SPIR-V id (made on first use; arrays and structs: the plain one)
static uint32_t tid(Sp *S, Type *t){
    if (t->id) return t->id;
    switch (t->k){
    case T_VOID: t->id = new_id(S); OP(S, &S->globals, OpTypeVoid, t->id); break;
    case T_BOOL: t->id = new_id(S); OP(S, &S->globals, OpTypeBool, t->id); break;
    case T_I32: t->id = new_id(S); OP(S, &S->globals, OpTypeInt, t->id, 32, 1); break;
    case T_U32: t->id = new_id(S); OP(S, &S->globals, OpTypeInt, t->id, 32, 0); break;
    case T_F32: t->id = new_id(S); OP(S, &S->globals, OpTypeFloat, t->id, 32); break;
    case T_AINT: case T_AFLOAT: wgsl_fail(S->W, 0, "internal: an abstract type reached SPIR-V"); break;
    case T_VEC: { uint32_t e = tid(S, t->e); t->id = new_id(S); OP(S, &S->globals, OpTypeVector, t->id, e, (uint32_t)t->n); break; }
    case T_MAT: { uint32_t e = tid(S, t->e); t->id = new_id(S); OP(S, &S->globals, OpTypeMatrix, t->id, e, (uint32_t)t->n); break; }
    case T_ATOMIC: t->id = tid(S, t->e); break;
    case T_ARRAY: case T_STRUCT: t->id = composite(S, t, 0); break;
    case T_RTARRAY: t->id = tid_l(S, t); break;
    case T_PTR: {
        if (t->e->k == T_ATOMIC){ t->id = tid(S, t_ptr(S, t->e->e, t->sc)); break; }   // (the same SPIR-V type)
        uint32_t e = buffer_sc(t->sc) ? tid_l(S, t->e) : tid(S, t->e);
        t->id = new_id(S);
        OP(S, &S->globals, OpTypePointer, t->id, (uint32_t)t->sc, e);
        break;
    }
    case T_TEX: case T_TEXDEPTH: {
        uint32_t f = tid(S, t_f32(S));
        t->id = new_id(S);
        // sampled type, Dim 2D (1), depth, arrayed, multisampled, sampled (1: with a sampler or fetch), format Unknown
        OP(S, &S->globals, OpTypeImage, t->id, f, 1, t->k == T_TEXDEPTH ? 1u : 0u, 0, 0, 1, 0);
        break;
    }
    }
    return t->id;
}

// ---- constants ----
static uint32_t const_bits(Sp *S, Type *t, uint64_t bits){
    for (int i = 0; i < S->nconsts; i++) if (S->consts[i].t == t && S->consts[i].bits == bits) return S->consts[i].id;
    uint32_t ty = tid(S, t), id = new_id(S);
    if (t->k == T_BOOL) OP(S, &S->globals, bits ? OpConstantTrue : OpConstantFalse, ty, id);
    else OP(S, &S->globals, OpConstant, ty, id, (uint32_t)bits);
    if (S->nconsts == S->cap_consts){
        int cap = S->cap_consts ? 2 * S->cap_consts : 64;
        void *v = realloc(S->consts, (size_t)cap * sizeof(*S->consts));
        if (!v) wgsl_fail(S->W, 0, "out of memory");
        S->consts = v; S->cap_consts = cap;
    }
    S->consts[S->nconsts].t = t; S->consts[S->nconsts].bits = bits; S->consts[S->nconsts].id = id;
    S->nconsts++;
    return id;
}
static uint32_t c_u32(Sp *S, uint32_t v){ return const_bits(S, t_u32(S), v); }
static uint32_t c_null(Sp *S, Type *t){
    // a composite's zero (OpConstantNull, kept with the scalar constants under bits ~0)
    for (int i = 0; i < S->nconsts; i++) if (S->consts[i].t == t && S->consts[i].bits == ~0ull) return S->consts[i].id;
    uint32_t ty = tid(S, t), id = new_id(S);
    OP(S, &S->globals, OpConstantNull, ty, id);
    if (S->nconsts == S->cap_consts){
        int cap = S->cap_consts ? 2 * S->cap_consts : 64;
        void *v = realloc(S->consts, (size_t)cap * sizeof(*S->consts));
        if (!v) wgsl_fail(S->W, 0, "out of memory");
        S->consts = v; S->cap_consts = cap;
    }
    S->consts[S->nconsts].t = t; S->consts[S->nconsts].bits = ~0ull; S->consts[S->nconsts].id = id;
    S->nconsts++;
    return id;
}

// ---- values ----
#define MAXC 16
typedef struct {
    Type *t;                         // its type (a reference: the type it refers to)
    uint32_t id;                     // the value, or (a reference) the pointer; 0: a constant not made yet
    int ref, sc;                     // a reference: a pointer into storage class sc
    int has_cv;                      // a constant (abstract, or concrete and known)
    double cv[MAXC];                 // its components
} Val;

// ---- symbols ----
enum { SYM_VALUE, SYM_VAR, SYM_GLOBAL, SYM_CONST, SYM_FN, SYM_STRUCT };
typedef struct {
    const char *name; int nlen;
    int kind;
    Val v;                           // values, variables (a reference), constants
    int index;                       // globals, functions, structs: their index
} Sym;

typedef struct {
    FnDecl *d;
    uint8_t *uses, *calls;           // the globals it uses, the functions it calls
    uint32_t *io; int nio;           // an entry point's inputs and outputs
    uint32_t id, type_id;
    Type *ret;
    Type **pt;                       // the parameters' types
    int resolved, emitted;
} FnInfo;

typedef struct {
    Type *t;                         // its type (storage/uniform: the wrapper struct's member)
    uint32_t var;                    // the variable
    int sc;
    int wrapped;                     // a buffer: the variable is a struct whose member 0 is the WGSL variable
    int resolved;
    Val cv;                          // a const's value
} GlobalInfo;

typedef struct {
    Sp sp;
    Wgsl *W;
    Sym *syms; int nsyms, cap_syms;
    int *scopes; int nscopes;
    FnInfo *fns;
    GlobalInfo *gl;
    StructInfo **structs;
    // the function being emitted
    FnInfo *fn;
    uint32_t block;                  // the current block's label
    int dead;                        // the current block has ended (a branch, return)
    struct { uint32_t merge, cont; } loops[32]; int nloops;
    // entry point interfaces
    uint32_t *iface; int niface, cap_iface;
    struct { uint32_t var; Type *t; } outs[16]; int nouts;   // the entry point's outputs, in order
    int out_is_struct;
    int stage;
    int uses_frag_depth;
    struct { Type *ret; Type *pt[16]; int n; uint32_t id; } fnty[256]; int nfnty;   // function types made
} Cx;

static Sp *SP(Cx *c){ return &c->sp; }

static void sym_push(Cx *c, Sym s){
    if (c->nsyms == c->cap_syms){
        int cap = c->cap_syms ? 2 * c->cap_syms : 256;
        Sym *v = realloc(c->syms, (size_t)cap * sizeof(Sym));
        if (!v) wgsl_fail(c->W, 0, "out of memory");
        c->syms = v; c->cap_syms = cap;
    }
    c->syms[c->nsyms++] = s;
}
static void scope_open(Cx *c){
    int *v = realloc(c->scopes, (size_t)(c->nscopes + 1) * sizeof(int));
    if (!v) wgsl_fail(c->W, 0, "out of memory");
    c->scopes = v;
    c->scopes[c->nscopes++] = c->nsyms;
}
static void scope_close(Cx *c){ c->nsyms = c->scopes[--c->nscopes]; }

static Sym *lookup(Cx *c, const char *name, int nlen){
    for (int i = c->nsyms - 1; i >= 0; i--)
        if (c->syms[i].nlen == nlen && !memcmp(c->syms[i].name, name, (size_t)nlen)) return &c->syms[i];
    return NULL;
}

static Val eval(Cx *c, Node *e);
static Type *resolve_type(Cx *c, Node *t);
static void resolve_global(Cx *c, int i);

// ---- structs ----
static Type *resolve_struct(Cx *c, int i){
    StructInfo *si = c->structs[i];
    StructDecl *d = si->decl;
    if (si->n || !d->n) { if (!si->n && !d->n) wgsl_fail(c->W, d->line, "an empty struct"); return mk(SP(c), T_STRUCT, NULL, 0, 0, si); }
    if (si->resolving) wgsl_fail(c->W, d->line, "struct %.*s contains itself", d->nlen, d->name);
    si->resolving = 1;
    int n = d->n;
    si->mname = wgsl_alloc(c->W, (size_t)n * sizeof(char *));
    si->mlen = wgsl_alloc(c->W, (size_t)n * sizeof(int));
    si->mt = wgsl_alloc(c->W, (size_t)n * sizeof(Type *));
    si->off = wgsl_alloc(c->W, (size_t)n * sizeof(uint32_t));
    si->mat = wgsl_alloc(c->W, (size_t)n * sizeof(Attrs));
    uint32_t at = 0, al = 4;
    for (int k = 0; k < n; k++){
        si->mname[k] = d->f[k].name; si->mlen[k] = d->f[k].nlen;
        si->mt[k] = resolve_type(c, d->f[k].type);
        si->mat[k] = d->f[k].at;
        uint32_t a = align_of(si->mt[k]);
        at = round_up(a, at);
        si->off[k] = at;
        at += size_of(si->mt[k]);
        if (a > al) al = a;
    }
    si->align = al;
    si->size = round_up(al, at);
    si->n = n;
    si->resolving = 0;
    return mk(SP(c), T_STRUCT, NULL, 0, 0, si);
}

// ---- types written in the source ----
static int const_int(Cx *c, Node *e, int line){
    Val v = eval(c, e);
    if (!v.has_cv || !is_int(v.t)) wgsl_fail(c->W, line, "expected a constant integer");
    return (int)v.cv[0];
}

static Type *resolve_type(Cx *c, Node *n){
    Sp *S = SP(c);
    if (!n) return t_void(S);
    const char *s = n->name; int l = n->nlen;
#define IS(x) wgsl_name_is(s, l, x)
    if (n->k == N_IDENT || (n->k == N_TYPE && !n->n)){
        if (IS("bool")) return t_bool(S);
        if (IS("i32")) return t_i32(S);
        if (IS("u32")) return t_u32(S);
        if (IS("f32")) return t_f32(S);
        if (IS("f16")) wgsl_fail(c->W, n->line, "f16 is not supported");
        if (IS("vec2f")) return t_vec(S, t_f32(S), 2);
        if (IS("vec3f")) return t_vec(S, t_f32(S), 3);
        if (IS("vec4f")) return t_vec(S, t_f32(S), 4);
        if (IS("vec2u")) return t_vec(S, t_u32(S), 2);
        if (IS("vec3u")) return t_vec(S, t_u32(S), 3);
        if (IS("vec4u")) return t_vec(S, t_u32(S), 4);
        if (IS("vec2i")) return t_vec(S, t_i32(S), 2);
        if (IS("vec3i")) return t_vec(S, t_i32(S), 3);
        if (IS("vec4i")) return t_vec(S, t_i32(S), 4);
        if (l == 7 && s[0] == 'm' && s[1] == 'a' && s[2] == 't' && s[4] == 'x' && s[6] == 'f' &&
            s[3] >= '2' && s[3] <= '4' && s[5] >= '2' && s[5] <= '4') return t_mat(S, s[3] - '0', s[5] - '0');
        if (IS("texture_depth_2d")) return mk(S, T_TEXDEPTH, NULL, 0, 0, NULL);
        Sym *sym = lookup(c, s, l);
        if (sym && sym->kind == SYM_STRUCT) return resolve_struct(c, sym->index);
        wgsl_fail(c->W, n->line, "unknown type '%.*s'", l, s);
    }
    // templated: vecN<T>, matCxR<f32>, array<T[, N]>, atomic<T>, ptr<space, T[, access]>, texture_2d<f32>
    if (IS("vec2") || IS("vec3") || IS("vec4")) return t_vec(S, resolve_type(c, n->list[0]), s[3] - '0');
    if (l == 6 && s[0] == 'm' && s[4] == 'x') return t_mat(S, s[3] - '0', s[5] - '0');
    if (IS("array")){
        Type *e = resolve_type(c, n->list[0]);
        if (n->n == 1) return mk(S, T_RTARRAY, e, 0, 0, NULL);
        int len = const_int(c, n->list[1], n->line);
        if (len <= 0) wgsl_fail(c->W, n->line, "an array's length must be positive");
        return mk(S, T_ARRAY, e, len, 0, NULL);
    }
    if (IS("atomic")){
        Type *e = resolve_type(c, n->list[0]);
        if (e->k != T_U32 && e->k != T_I32) wgsl_fail(c->W, n->line, "atomic<T>: T must be u32 or i32");
        return mk(S, T_ATOMIC, e, 0, 0, NULL);
    }
    if (IS("ptr")){
        Node *sp = n->list[0];
        int sc = wgsl_name_is(sp->name, sp->nlen, "function") ? SC_Function : wgsl_name_is(sp->name, sp->nlen, "private") ? SC_Private :
                 wgsl_name_is(sp->name, sp->nlen, "workgroup") ? SC_Workgroup : wgsl_name_is(sp->name, sp->nlen, "storage") ? SC_StorageBuffer :
                 wgsl_name_is(sp->name, sp->nlen, "uniform") ? SC_Uniform : -1;
        if (sc < 0) wgsl_fail(c->W, n->line, "a pointer's address space");
        return t_ptr(S, resolve_type(c, n->list[1]), sc);
    }
    if (IS("texture_2d")) return mk(S, T_TEX, NULL, 0, 0, NULL);
#undef IS
    wgsl_fail(c->W, n->line, "unknown type '%.*s'", l, s);
    return NULL;
}

// ---- emitting into the function ----
static Words *FB(Cx *c){ return &c->sp.fbody; }
static uint32_t nid(Cx *c){ return new_id(SP(c)); }
#define E(c, opc, ...) OP(SP(c), FB(c), opc, __VA_ARGS__)

static uint32_t ext(Cx *c, Type *t, uint32_t inst, int n, const uint32_t *args){
    uint32_t r = nid(c);
    uint32_t a[8];
    a[0] = tid(SP(c), t); a[1] = r; a[2] = SP(c)->glsl; a[3] = inst;
    for (int i = 0; i < n; i++) a[4 + i] = args[i];
    op(SP(c), FB(c), OpExtInst, 4 + n, a);
    return r;
}

// ---- constants: normalized to their type, folded, made when used ----
static double norm(Type *t, double x){
    switch (scalar_of(t)->k){
    case T_U32: { double m = fmod(trunc(x), 4294967296.0); return m < 0 ? m + 4294967296.0 : m; }
    case T_I32: { double m = fmod(trunc(x), 4294967296.0); if (m < 0) m += 4294967296.0; return m >= 2147483648.0 ? m - 4294967296.0 : m; }
    case T_F32: return (double)(float)x;
    case T_BOOL: return x != 0;
    default: return x;
    }
}

static uint64_t scalar_bits(Type *t, double x){
    switch (t->k){
    case T_BOOL: return x != 0;
    case T_I32: return (uint32_t)(int32_t)x;
    case T_U32: return (uint32_t)x;
    case T_F32: { float f = (float)x; uint32_t b; memcpy(&b, &f, 4); return b; }
    default: return 0;
    }
}

static uint32_t const_id(Cx *c, Type *t, const double *cv){
    Sp *S = SP(c);
    if (is_scalar(t)) return const_bits(S, t, scalar_bits(t, cv[0]));
    if (t->k != T_VEC) wgsl_fail(c->W, 0, "internal: a constant of this type");
    for (int i = 0; i < S->ncc; i++)
        if (S->cc[i].t == t && !memcmp(S->cc[i].v, cv, sizeof(double) * (size_t)t->n)) return S->cc[i].id;
    uint32_t comp[4];
    for (int i = 0; i < t->n; i++) comp[i] = const_bits(S, t->e, scalar_bits(t->e, cv[i]));
    uint32_t ty = tid(S, t), id = new_id(S);
    uint32_t a[6] = {ty, id, comp[0], comp[1], t->n > 2 ? comp[2] : 0, t->n > 3 ? comp[3] : 0};
    op(S, &S->globals, OpConstantComposite, 2 + t->n, a);
    if (S->ncc == S->cap_cc){
        int cap = S->cap_cc ? 2 * S->cap_cc : 64;
        void *v = realloc(S->cc, (size_t)cap * sizeof(*S->cc));
        if (!v) wgsl_fail(c->W, 0, "out of memory");
        S->cc = v; S->cap_cc = cap;
    }
    S->cc[S->ncc].t = t; memcpy(S->cc[S->ncc].v, cv, sizeof(double) * (size_t)t->n); S->cc[S->ncc].id = id; S->ncc++;
    return id;
}

static Type *concrete(Cx *c, Type *t){
    Sp *S = SP(c);
    Type *s = scalar_of(t);
    if (s->k == T_AINT) return t_vec(S, t_i32(S), width_of(t));
    if (s->k == T_AFLOAT) return t_vec(S, t_f32(S), width_of(t));
    return t;
}

// v as type want (an abstract constant to a concrete type of the same shape; else as it is)
static Val as_type(Cx *c, Val v, Type *want, int line){
    if (v.t == want) return v;
    if (is_abstract(v.t) && v.has_cv && (is_scalar(want) || want->k == T_VEC)){
        Type *ws = scalar_of(want);
        int wn = width_of(want), vn = width_of(v.t);
        if (vn != wn && vn != 1) goto bad;
        if (scalar_of(v.t)->k == T_AFLOAT && ws->k != T_F32 && ws->k != T_AFLOAT) goto bad;
        if (ws->k == T_BOOL) goto bad;
        Val r = v;
        r.t = want;
        r.id = 0;
        for (int i = 0; i < wn; i++) r.cv[i] = norm(want, v.cv[vn == 1 ? 0 : i]);
        return r;
    }
bad:;
    char a[64], b[64];
    type_name(v.t, a, sizeof(a)); type_name(want, b, sizeof(b));
    wgsl_fail(c->W, line, "expected %s, got %s", b, a);
    return v;
}

static uint32_t load(Cx *c, Val *v);

// a value's id: constants made, references loaded
static uint32_t value(Cx *c, Val *v){
    if (v->ref) return load(c, v);
    if (!v->id){
        if (!v->has_cv) wgsl_fail(c->W, 0, "internal: a value without an id");
        if (is_abstract(v->t)) *v = as_type(c, *v, concrete(c, v->t), 0);
        v->id = const_id(c, v->t, v->cv);
    }
    return v->id;
}

static uint32_t load(Cx *c, Val *v){
    if (!v->ref) return value(c, v);
    if (v->t->k == T_ATOMIC) wgsl_fail(c->W, 0, "an atomic must be read with atomicLoad");
    uint32_t r = nid(c);
    if (buffer_sc(v->sc) && layout_differs(v->t)){
        // (a buffer's array or struct: read with its layout, then as the plain type)
        uint32_t x = nid(c);
        E(c, OpLoad, tid_l(SP(c), v->t), x, v->id);
        E(c, OpCopyLogical, tid(SP(c), v->t), r, x);
    } else E(c, OpLoad, tid(SP(c), v->t), r, v->id);
    Val x;
    memset(&x, 0, sizeof(x));
    x.t = v->t; x.id = r;
    *v = x;
    return r;
}

static Val mkval(Type *t, uint32_t id){ Val v; memset(&v, 0, sizeof(v)); v.t = t; v.id = id; return v; }
static Val mkconst(Type *t, double x){ Val v; memset(&v, 0, sizeof(v)); v.t = t; v.has_cv = 1; for (int i = 0; i < width_of(t); i++) v.cv[i] = x; return v; }

// a scalar as a vector of n (a constant stays one)
static Val splat(Cx *c, Val s, int n){
    Sp *S = SP(c);
    if (n == 1 || s.t->k == T_VEC) return s;
    if (s.has_cv && !s.ref){
        Val r = s;
        r.t = t_vec(S, s.t, n); r.id = 0;
        for (int i = 1; i < n; i++) r.cv[i] = s.cv[0];
        return r;
    }
    uint32_t x = value(c, &s), r = nid(c);
    Type *vt = t_vec(S, s.t, n);
    uint32_t a[6] = {tid(S, vt), r, x, x, x, x};
    op(S, FB(c), OpCompositeConstruct, 2 + n, a);
    return mkval(vt, r);
}

// two operands to one scalar type (abstract ones follow the concrete one; both abstract: float wins)
static void unify(Cx *c, Val *a, Val *b, int line){
    Sp *S = SP(c);
    int aa = is_abstract(a->t), ba = is_abstract(b->t);
    if (aa && !ba){ Type *s = scalar_of(b->t); *a = as_type(c, *a, t_vec(S, s, width_of(a->t)), line); }
    else if (ba && !aa){ Type *s = scalar_of(a->t); *b = as_type(c, *b, t_vec(S, s, width_of(b->t)), line); }
    else if (aa && ba && scalar_of(a->t)->k != scalar_of(b->t)->k){
        if (scalar_of(a->t)->k == T_AINT) a->t = t_vec(S, t_afloat(S), width_of(a->t));
        else b->t = t_vec(S, t_afloat(S), width_of(b->t));
    }
}

// ---- operators ----
static double fold1(int op, Type *t, double a, double b, int *ok){
    *ok = 1;
    int f = is_float(t), u = is_uint(t);
    uint64_t ua = (uint64_t)(int64_t)a & 0xffffffffull, ub = (uint64_t)(int64_t)b & 0xffffffffull;
    switch (op){
    case TK_PLUS: return a + b;
    case TK_MINUS: return a - b;
    case TK_STAR: return a * b;
    case TK_SLASH: if (f) return a / b; if (b == 0){ *ok = 0; return 0; } return trunc(a / b);
    case TK_PERCENT: if (f) return fmod(a, b); if (b == 0){ *ok = 0; return 0; } return fmod(a, b);
    case TK_AMP: return (double)(u ? (ua & ub) : (uint64_t)((int64_t)a & (int64_t)b));
    case TK_PIPE: return (double)(u ? (ua | ub) : (uint64_t)((int64_t)a | (int64_t)b));
    case TK_CARET: return (double)(u ? (ua ^ ub) : (uint64_t)((int64_t)a ^ (int64_t)b));
    case TK_SHL: return (double)(((uint64_t)(int64_t)a << ((int)b & 31)) & (is_abstract(t) ? ~0ull : 0xffffffffull));
    case TK_SHR: return u ? (double)(ua >> ((int)b & 31)) : (double)((int64_t)a >> ((int)b & 31));
    default: *ok = 0; return 0;
    }
}

static int is_compare(int op){ return op == TK_LT || op == TK_GT || op == TK_LE || op == TK_GE || op == TK_EQEQ || op == TK_NE; }

static Val binop(Cx *c, int opk, Val a, Val b, int line){
    Sp *S = SP(c);
    if (a.ref) load(c, &a);
    if (b.ref) load(c, &b);
    // shifts: the amount is u32 (or abstract), the result the left side's type
    if (opk == TK_SHL || opk == TK_SHR){
        if (is_abstract(b.t)) b = as_type(c, b, t_vec(S, t_u32(S), width_of(b.t)), line);
        if (is_abstract(a.t) && !(a.has_cv && b.has_cv)) a = as_type(c, a, concrete(c, a.t), line);
    } else unify(c, &a, &b, line);
    // scalar with vector: the scalar spread (not for matrix products)
    if (a.t->k != T_MAT && b.t->k != T_MAT && width_of(a.t) != width_of(b.t)){
        if (opk == TK_STAR && is_float(a.t) && !is_abstract(a.t) && !(a.has_cv && b.has_cv)){
            // float vector * scalar: OpVectorTimesScalar below
        } else if (width_of(a.t) == 1) a = splat(c, a, width_of(b.t));
        else if (width_of(b.t) == 1) b = splat(c, b, width_of(a.t));
    }
    // constants fold
    if (a.has_cv && b.has_cv && !a.ref && !b.ref && (is_scalar(a.t) || a.t->k == T_VEC) && width_of(a.t) == width_of(b.t)){
        int n = width_of(a.t), ok = 1;
        Val r;
        memset(&r, 0, sizeof(r));
        r.has_cv = 1;
        if (is_compare(opk)){
            r.t = t_vec(S, t_bool(S), n);
            for (int i = 0; i < n; i++){
                double x = a.cv[i], y = b.cv[i];
                r.cv[i] = opk == TK_LT ? x < y : opk == TK_GT ? x > y : opk == TK_LE ? x <= y : opk == TK_GE ? x >= y : opk == TK_EQEQ ? x == y : x != y;
            }
            return r;
        }
        if (opk == TK_ANDAND || opk == TK_OROR){
            r.t = a.t;
            for (int i = 0; i < n; i++) r.cv[i] = opk == TK_ANDAND ? (a.cv[i] && b.cv[i]) : (a.cv[i] || b.cv[i]);
            return r;
        }
        r.t = a.t;
        for (int i = 0; i < n && ok; i++) r.cv[i] = norm(a.t, fold1(opk, a.t, a.cv[i], b.cv[i], &ok));
        if (ok) return r;
    }
    if (is_abstract(a.t)) a = as_type(c, a, concrete(c, a.t), line);
    if (is_abstract(b.t)) b = as_type(c, b, concrete(c, b.t), line);
    uint32_t x = value(c, &a), y = value(c, &b), r = nid(c);
    Type *t = a.t;
    // matrices
    if (a.t->k == T_MAT || b.t->k == T_MAT){
        if (opk != TK_STAR) wgsl_fail(c->W, line, "only * is supported on matrices");
        Type *rt;
        uint32_t o;
        if (a.t->k == T_MAT && b.t->k == T_MAT){ o = OpMatrixTimesMatrix; rt = t_mat(S, b.t->n, a.t->e->n); }
        else if (a.t->k == T_MAT && b.t->k == T_VEC){ o = OpMatrixTimesVector; rt = a.t->e; }
        else if (a.t->k == T_VEC){ o = OpVectorTimesMatrix; rt = t_vec(S, t_f32(S), b.t->n); }
        else if (a.t->k == T_MAT){ o = OpMatrixTimesScalar; rt = a.t; }
        else { o = OpMatrixTimesScalar; rt = b.t; uint32_t z = x; x = y; y = z; }
        E(c, o, tid(S, rt), r, x, y);
        return mkval(rt, r);
    }
    if (width_of(a.t) != width_of(b.t)){
        // float vector * scalar
        uint32_t vec = width_of(a.t) > 1 ? x : y, sc = width_of(a.t) > 1 ? y : x;
        Type *vt = width_of(a.t) > 1 ? a.t : b.t;
        E(c, OpVectorTimesScalar, tid(S, vt), r, vec, sc);
        return mkval(vt, r);
    }
    if (scalar_of(a.t) != scalar_of(b.t)){
        char p[64], q[64];
        type_name(a.t, p, sizeof(p)); type_name(b.t, q, sizeof(q));
        wgsl_fail(c->W, line, "mixed types: %s and %s", p, q);
    }
    int f = is_float(t), s = is_sint(t), bl = is_bool(t);
    uint32_t o = 0;
    Type *rt = t;
    switch (opk){
    case TK_PLUS: o = f ? OpFAdd : OpIAdd; break;
    case TK_MINUS: o = f ? OpFSub : OpISub; break;
    case TK_STAR: o = f ? OpFMul : OpIMul; break;
    case TK_SLASH: o = f ? OpFDiv : s ? OpSDiv : OpUDiv; break;
    case TK_PERCENT: o = f ? OpFRem : s ? OpSRem : OpUMod; break;
    case TK_AMP: o = bl ? OpLogicalAnd : OpBitwiseAnd; break;
    case TK_PIPE: o = bl ? OpLogicalOr : OpBitwiseOr; break;
    case TK_CARET: o = bl ? OpLogicalNotEqual : OpBitwiseXor; break;
    case TK_ANDAND: o = OpLogicalAnd; break;
    case TK_OROR: o = OpLogicalOr; break;
    case TK_SHL: case TK_SHR: {
        // WGSL takes the amount modulo the bit width
        if (!b.has_cv){
            Val m = splat(c, mkconst(t_u32(S), 31), width_of(b.t));
            uint32_t mm = value(c, &m), yy = nid(c);
            E(c, OpBitwiseAnd, tid(S, b.t), yy, y, mm);
            y = yy;
        }
        o = opk == TK_SHL ? OpShiftLeftLogical : s ? OpShiftRightArithmetic : OpShiftRightLogical;
        break;
    }
    default:
        rt = t_vec(S, t_bool(S), width_of(t));
        if (bl) o = opk == TK_EQEQ ? OpLogicalEqual : opk == TK_NE ? OpLogicalNotEqual : 0;
        else if (f) o = opk == TK_LT ? OpFOrdLessThan : opk == TK_GT ? OpFOrdGreaterThan : opk == TK_LE ? OpFOrdLessThanEqual :
                        opk == TK_GE ? OpFOrdGreaterThanEqual : opk == TK_EQEQ ? OpFOrdEqual : OpFUnordNotEqual;
        else if (s) o = opk == TK_LT ? OpSLessThan : opk == TK_GT ? OpSGreaterThan : opk == TK_LE ? OpSLessThanEqual :
                        opk == TK_GE ? OpSGreaterThanEqual : opk == TK_EQEQ ? OpIEqual : OpINotEqual;
        else o = opk == TK_LT ? OpULessThan : opk == TK_GT ? OpUGreaterThan : opk == TK_LE ? OpULessThanEqual :
                 opk == TK_GE ? OpUGreaterThanEqual : opk == TK_EQEQ ? OpIEqual : OpINotEqual;
    }
    if (!o) wgsl_fail(c->W, line, "this operator does not apply to these operands");
    if ((bl && (opk == TK_PLUS || opk == TK_MINUS || opk == TK_STAR || opk == TK_SLASH || opk == TK_PERCENT)) ||
        (f && (opk == TK_AMP || opk == TK_PIPE || opk == TK_CARET || opk == TK_SHL || opk == TK_SHR)))
        wgsl_fail(c->W, line, "this operator does not apply to these operands");
    E(c, o, tid(S, rt), r, x, y);
    return mkval(rt, r);
}

static Val unop(Cx *c, int opk, Val a, int line){
    Sp *S = SP(c);
    if (a.ref) load(c, &a);
    if (a.has_cv){
        Val r = a;
        r.id = 0;
        for (int i = 0; i < width_of(a.t); i++){
            double x = a.cv[i];
            r.cv[i] = norm(a.t, opk == TK_MINUS ? -x : opk == TK_BANG ? !x :
                               (double)(is_uint(a.t) ? (uint64_t)(~(uint32_t)(int64_t)x) : (uint64_t)~(int64_t)x));
        }
        if (opk == TK_TILDE && is_sint(a.t)) for (int i = 0; i < width_of(a.t); i++) r.cv[i] = (double)~(int64_t)a.cv[i];
        return r;
    }
    uint32_t x = value(c, &a), r = nid(c);
    uint32_t o = opk == TK_MINUS ? (is_float(a.t) ? OpFNegate : OpSNegate) : opk == TK_BANG ? OpLogicalNot : OpNot;
    if ((opk == TK_BANG && !is_bool(a.t)) || (opk == TK_TILDE && !is_int(a.t)) || (opk == TK_MINUS && (is_bool(a.t) || a.t->k == T_MAT)))
        wgsl_fail(c->W, line, "this operator does not apply here");
    E(c, o, tid(S, a.t), r, x);
    return mkval(a.t, r);
}

// ---- blocks ----
static void label(Cx *c, uint32_t id){ E(c, OpLabel, id); c->block = id; c->dead = 0; }
static void branch(Cx *c, uint32_t to){ if (!c->dead){ E(c, OpBranch, to); c->dead = 1; } }

// a function variable (declared in the function's first block, as SPIR-V wants)
static uint32_t fvar(Cx *c, Type *t, uint32_t init){
    Sp *S = SP(c);
    uint32_t p = tid(S, t_ptr(S, t, SC_Function)), r = new_id(S);
    if (init) OP(S, &S->fvars, OpVariable, p, r, SC_Function, init);
    else OP(S, &S->fvars, OpVariable, p, r, SC_Function);
    return r;
}

// ---- members, swizzles, indexing ----
static int swz(char ch){
    switch (ch){ case 'x': case 'r': return 0; case 'y': case 'g': return 1; case 'z': case 'b': return 2; case 'w': case 'a': return 3; }
    return -1;
}

static Val as_ref(Val v){
    // a pointer value used as what it points to (WGSL lets p.x and p[i] mean (*p).x, (*p)[i])
    if (!v.ref && v.t->k == T_PTR){ Val r; memset(&r, 0, sizeof(r)); r.t = v.t->e; r.id = v.id; r.ref = 1; r.sc = v.t->sc; return r; }
    return v;
}

static Val member(Cx *c, Val v, const char *name, int nlen, int line){
    Sp *S = SP(c);
    v = as_ref(v);
    if (v.t->k == T_STRUCT){
        StructInfo *si = v.t->s;
        int k = -1;
        for (int i = 0; i < si->n; i++) if (si->mlen[i] == nlen && !memcmp(si->mname[i], name, (size_t)nlen)) k = i;
        if (k < 0) wgsl_fail(c->W, line, "%.*s has no member %.*s", si->nlen, si->name, nlen, name);
        Type *mt = si->mt[k];
        uint32_t r = nid(c);
        if (v.ref){
            E(c, OpAccessChain, tid(S, t_ptr(S, mt, v.sc)), r, v.id, c_u32(S, (uint32_t)k));
            Val x; memset(&x, 0, sizeof(x)); x.t = mt; x.id = r; x.ref = 1; x.sc = v.sc;
            return x;
        }
        E(c, OpCompositeExtract, tid(S, mt), r, value(c, &v), (uint32_t)k);
        return mkval(mt, r);
    }
    if (v.t->k == T_VEC){
        int idx[4];
        if (nlen > 4) goto bad;
        for (int i = 0; i < nlen; i++){ idx[i] = swz(name[i]); if (idx[i] < 0 || idx[i] >= v.t->n) goto bad; }
        Type *e = v.t->e;
        if (nlen == 1){
            if (v.ref){
                uint32_t r = nid(c);
                E(c, OpAccessChain, tid(S, t_ptr(S, e, v.sc)), r, v.id, c_u32(S, (uint32_t)idx[0]));
                Val x; memset(&x, 0, sizeof(x)); x.t = e; x.id = r; x.ref = 1; x.sc = v.sc;
                return x;
            }
            if (v.has_cv){ Val x = mkconst(e, v.cv[idx[0]]); return x; }
            uint32_t r = nid(c);
            E(c, OpCompositeExtract, tid(S, e), r, value(c, &v), (uint32_t)idx[0]);
            return mkval(e, r);
        }
        Type *rt = t_vec(S, e, nlen);
        if (v.has_cv && !v.ref){ Val x; memset(&x, 0, sizeof(x)); x.t = rt; x.has_cv = 1; for (int i = 0; i < nlen; i++) x.cv[i] = v.cv[idx[i]]; return x; }
        uint32_t src = load(c, &v), r = nid(c);
        uint32_t a[8] = {tid(S, rt), r, src, src, (uint32_t)idx[0], (uint32_t)idx[1], nlen > 2 ? (uint32_t)idx[2] : 0, nlen > 3 ? (uint32_t)idx[3] : 0};
        op(S, FB(c), OpVectorShuffle, 4 + nlen, a);
        return mkval(rt, r);
    }
bad:
    wgsl_fail(c->W, line, "no member .%.*s here", nlen, name);
    return v;
}

static Val index_of(Cx *c, Val v, Val ix, int line){
    Sp *S = SP(c);
    v = as_ref(v);
    if (ix.ref) load(c, &ix);
    if (!is_int(ix.t) || width_of(ix.t) != 1) wgsl_fail(c->W, line, "an index must be an integer");
    Type *et;
    if (v.t->k == T_VEC) et = v.t->e;
    else if (v.t->k == T_MAT) et = v.t->e;
    else if (v.t->k == T_ARRAY || v.t->k == T_RTARRAY) et = v.t->e;
    else { wgsl_fail(c->W, line, "this cannot be indexed"); return v; }
    if (is_abstract(ix.t)) ix = as_type(c, ix, t_i32(S), line);
    if (v.ref){
        uint32_t r = nid(c), i = value(c, &ix);
        E(c, OpAccessChain, tid(S, t_ptr(S, et, v.sc)), r, v.id, i);
        Val x; memset(&x, 0, sizeof(x)); x.t = et; x.id = r; x.ref = 1; x.sc = v.sc;
        return x;
    }
    if (ix.has_cv){
        if (v.has_cv && v.t->k == T_VEC) return mkconst(et, v.cv[(int)ix.cv[0]]);
        uint32_t r = nid(c);
        E(c, OpCompositeExtract, tid(S, et), r, value(c, &v), (uint32_t)ix.cv[0]);
        return mkval(et, r);
    }
    if (v.t->k == T_VEC){
        uint32_t r = nid(c);
        E(c, OpVectorExtractDynamic, tid(S, et), r, value(c, &v), value(c, &ix));
        return mkval(et, r);
    }
    // a value indexed at run time: through a function variable
    uint32_t var = fvar(c, v.t, 0);
    E(c, OpStore, var, value(c, &v));
    Val ref; memset(&ref, 0, sizeof(ref)); ref.t = v.t; ref.id = var; ref.ref = 1; ref.sc = SC_Function;
    Val x = index_of(c, ref, ix, line);
    load(c, &x);
    return x;
}

// ---- conversions and constructors ----
static Val convert(Cx *c, Val v, Type *to, int line){
    Sp *S = SP(c);
    if (v.ref) load(c, &v);
    if (v.t == to) return v;
    if (width_of(v.t) != width_of(to) || !(is_scalar(scalar_of(v.t)) && is_scalar(scalar_of(to)))){
        if (is_abstract(v.t)) return as_type(c, v, to, line);
        char a[64], b[64];
        type_name(v.t, a, sizeof(a)); type_name(to, b, sizeof(b));
        wgsl_fail(c->W, line, "cannot convert %s to %s", a, b);
    }
    Type *fs = scalar_of(v.t), *ts = scalar_of(to);
    if (v.has_cv){
        Val r = v;
        r.t = to; r.id = 0;
        for (int i = 0; i < width_of(to); i++){
            double x = v.cv[i];
            if (ts->k == T_BOOL) r.cv[i] = x != 0;
            else if (fs->k == T_F32 || fs->k == T_AFLOAT) r.cv[i] = ts->k == T_F32 ? (double)(float)x : trunc(x);
            else if ((fs->k == T_I32 || fs->k == T_U32) && (ts->k == T_I32 || ts->k == T_U32)) r.cv[i] = norm(to, x);  // (bits kept)
            else r.cv[i] = norm(to, x);
        }
        return r;
    }
    uint32_t x = value(c, &v), r = nid(c), ty = tid(S, to);
    if (ts->k == T_BOOL){
        Val z = splat(c, mkconst(fs, 0), width_of(v.t));
        E(c, is_float(v.t) ? OpFUnordNotEqual : OpINotEqual, ty, r, x, value(c, &z));
        return mkval(to, r);
    }
    if (fs->k == T_BOOL){
        Val one = splat(c, mkconst(ts, 1), width_of(to)), zero = splat(c, mkconst(ts, 0), width_of(to));
        E(c, OpSelect, ty, r, x, value(c, &one), value(c, &zero));
        return mkval(to, r);
    }
    uint32_t o;
    if (fs->k == T_F32) o = ts->k == T_U32 ? OpConvertFToU : OpConvertFToS;
    else if (ts->k == T_F32) o = fs->k == T_U32 ? OpConvertUToF : OpConvertSToF;
    else o = OpBitcast;                                  // i32 <-> u32: the same bits
    E(c, o, ty, r, x);
    return mkval(to, r);
}

static Val construct(Cx *c, Type *t, Node *call){
    Sp *S = SP(c);
    int n = call->n, line = call->line;
    if (!n){
        if (is_scalar(t) || t->k == T_VEC) return mkconst(t, 0);
        return mkval(t, c_null(S, t));
    }
    if (is_scalar(t)){
        if (n != 1) wgsl_fail(c->W, line, "a scalar from one value");
        return convert(c, eval(c, call->list[0]), t, line);
    }
    Val args[16];
    if (n > 16) wgsl_fail(c->W, line, "too many arguments");
    for (int i = 0; i < n; i++){ args[i] = eval(c, call->list[i]); if (args[i].ref) load(c, &args[i]); }
    if (t->k == T_VEC){
        Type *e = t->e;
        if (n == 1 && width_of(args[0].t) == 1) return splat(c, convert(c, args[0], e, line), t->n);
        if (n == 1) return convert(c, args[0], t, line);
        // components and vectors, each to the element type
        int total = 0, all_const = 1;
        for (int i = 0; i < n; i++){
            args[i] = convert(c, args[i], t_vec(S, e, width_of(args[i].t)), line);
            total += width_of(args[i].t);
            if (!args[i].has_cv) all_const = 0;
        }
        if (total != t->n) wgsl_fail(c->W, line, "vec%d from %d components", t->n, total);
        if (all_const){
            Val r; memset(&r, 0, sizeof(r)); r.t = t; r.has_cv = 1;
            int k = 0;
            for (int i = 0; i < n; i++) for (int j = 0; j < width_of(args[i].t); j++) r.cv[k++] = args[i].cv[j];
            return r;
        }
        uint32_t a[6];
        a[0] = tid(S, t); a[1] = nid(c);
        for (int i = 0; i < n; i++) a[2 + i] = value(c, &args[i]);
        op(S, FB(c), OpCompositeConstruct, 2 + n, a);
        return mkval(t, a[1]);
    }
    if (t->k == T_MAT){
        Type *col = t->e;
        uint32_t cols[4];
        if (n == t->n){
            for (int i = 0; i < n; i++){ Val v = convert(c, args[i], col, line); cols[i] = value(c, &v); }
        } else if (n == t->n * col->n){
            for (int i = 0; i < t->n; i++){
                uint32_t a[6]; a[0] = tid(S, col); a[1] = nid(c);
                for (int j = 0; j < col->n; j++){ Val v = convert(c, args[i * col->n + j], t_f32(S), line); a[2 + j] = value(c, &v); }
                op(S, FB(c), OpCompositeConstruct, 2 + col->n, a);
                cols[i] = a[1];
            }
        } else wgsl_fail(c->W, line, "a matrix from %d values", n);
        uint32_t a[6]; a[0] = tid(S, t); a[1] = nid(c);
        for (int i = 0; i < t->n; i++) a[2 + i] = cols[i];
        op(S, FB(c), OpCompositeConstruct, 2 + t->n, a);
        return mkval(t, a[1]);
    }
    if (t->k == T_ARRAY || t->k == T_STRUCT){
        int want = t->k == T_ARRAY ? t->n : t->s->n;
        if (n != want) wgsl_fail(c->W, line, "%d values for %d elements", n, want);
        uint32_t *a = wgsl_alloc(c->W, (size_t)(2 + n) * sizeof(uint32_t));
        a[0] = tid(S, t); a[1] = nid(c);
        for (int i = 0; i < n; i++){
            Type *et = t->k == T_ARRAY ? t->e : t->s->mt[i];
            Val v = is_abstract(args[i].t) ? as_type(c, args[i], et, line) : args[i];
            if (v.t != et){ char p[64], q[64]; type_name(v.t, p, sizeof(p)); type_name(et, q, sizeof(q)); wgsl_fail(c->W, line, "expected %s, got %s", q, p); }
            a[2 + i] = value(c, &v);
        }
        op(S, FB(c), OpCompositeConstruct, 2 + n, a);
        return mkval(t, a[1]);
    }
    wgsl_fail(c->W, line, "cannot construct this type");
    return args[0];
}

// ---- built-in functions ----
static Val loaded(Cx *c, Node *e){ Val v = eval(c, e); if (v.ref) load(c, &v); return v; }

// arguments to one type (abstract ones follow the concrete ones; a scalar spread to the
// vectors' width where WGSL allows it: mix's t, clamp's bounds, ...)
static void common(Cx *c, Val *v, int n, int line){
    Sp *S = SP(c);
    Type *s = NULL;
    int w = 1;
    for (int i = 0; i < n; i++){
        if (!is_abstract(v[i].t) && (is_scalar(scalar_of(v[i].t)))) s = scalar_of(v[i].t);
        if (width_of(v[i].t) > w) w = width_of(v[i].t);
    }
    if (!s){
        int any_float = 0;
        for (int i = 0; i < n; i++) if (scalar_of(v[i].t)->k == T_AFLOAT) any_float = 1;
        s = any_float ? t_f32(S) : t_i32(S);
        for (int i = 0; i < n; i++) if (scalar_of(v[i].t)->k == T_AINT && any_float) v[i] = as_type(c, v[i], t_vec(S, t_f32(S), width_of(v[i].t)), line);
    }
    for (int i = 0; i < n; i++){
        if (is_abstract(v[i].t)) v[i] = as_type(c, v[i], t_vec(S, s, width_of(v[i].t)), line);
        if (width_of(v[i].t) != w) v[i] = splat(c, v[i], w);
    }
}

static Val gl(Cx *c, uint32_t inst, Val *v, int n, Type *rt){
    uint32_t a[4];
    for (int i = 0; i < n; i++) a[i] = value(c, &v[i]);
    return mkval(rt, ext(c, rt, inst, n, a));
}

static Val ptr_arg(Cx *c, Node *e, int line){
    Val p = eval(c, e);
    if (p.ref){ Val q; memset(&q, 0, sizeof(q)); q.t = t_ptr(SP(c), p.t, p.sc); q.id = p.id; return q; }
    if (p.t->k != T_PTR) wgsl_fail(c->W, line, "expected a pointer (&x)");
    return p;
}

static uint32_t barrier_scope_sem(Cx *c, uint32_t scope, uint32_t sem){ (void)c; return (scope << 16) | sem; }

static int builtin(Cx *c, Node *call, Val *out){
    Sp *S = SP(c);
    const char *s = call->name; int l = call->nlen, n = call->n, line = call->line;
#define IS(x) wgsl_name_is(s, l, x)
    Val v[4];
    // GLSL.std.450 one-to-one, float only
    static const struct { const char *name; uint32_t inst; int args; } fl[] = {
        {"floor", G450_Floor, 1}, {"ceil", G450_Ceil, 1}, {"fract", G450_Fract, 1}, {"round", 2, 1}, {"trunc", G450_Trunc, 1},
        {"sqrt", G450_Sqrt, 1}, {"inverseSqrt", G450_InverseSqrt, 1}, {"exp", G450_Exp, 1}, {"exp2", G450_Exp2, 1},
        {"log", G450_Log, 1}, {"log2", G450_Log2, 1}, {"sin", G450_Sin, 1}, {"cos", G450_Cos, 1}, {"tan", G450_Tan, 1},
        {"asin", G450_Asin, 1}, {"acos", G450_Acos, 1}, {"atan", G450_Atan, 1}, {"sinh", G450_Sinh, 1}, {"cosh", G450_Cosh, 1},
        {"tanh", G450_Tanh, 1}, {"atan2", G450_Atan2, 2}, {"pow", G450_Pow, 2}, {"fma", G450_Fma, 3}, {"step", G450_Step, 2},
        {"smoothstep", G450_SmoothStep, 3}, {"normalize", G450_Normalize, 1}, {"reflect", G450_Reflect, 2},
        {"faceForward", G450_FaceForward, 3}, {"degrees", 12, 1}, {"radians", 11, 1}};
    for (size_t i = 0; i < sizeof(fl) / sizeof(fl[0]); i++){
        if (!IS(fl[i].name)) continue;
        if (n != fl[i].args) wgsl_fail(c->W, line, "%s takes %d arguments", fl[i].name, fl[i].args);
        for (int k = 0; k < n; k++) v[k] = loaded(c, call->list[k]);
        common(c, v, n, line);
        if (!is_float(v[0].t)) wgsl_fail(c->W, line, "%s takes floats", fl[i].name);
        *out = gl(c, fl[i].inst, v, n, v[0].t);
        return 1;
    }
    if (IS("min") || IS("max") || IS("clamp") || IS("abs") || IS("sign") || IS("mix") || IS("saturate")){
        for (int k = 0; k < n && k < 3; k++) v[k] = loaded(c, call->list[k]);
        common(c, v, n, line);
        int f = is_float(v[0].t), sg = is_sint(v[0].t);
        uint32_t inst = 0;
        if (IS("min")) inst = f ? G450_FMin : sg ? G450_SMin : G450_UMin;
        else if (IS("max")) inst = f ? G450_FMax : sg ? G450_SMax : G450_UMax;
        else if (IS("clamp")) inst = f ? G450_FClamp : sg ? G450_SClamp : G450_UClamp;
        else if (IS("abs")){ if (is_uint(v[0].t)){ *out = v[0]; return 1; } inst = f ? G450_FAbs : G450_SAbs; }
        else if (IS("sign")) inst = f ? G450_FSign : G450_SSign;
        else if (IS("mix")) inst = G450_FMix;
        else {   // saturate: clamp(x, 0, 1)
            v[1] = splat(c, mkconst(scalar_of(v[0].t), 0), width_of(v[0].t));
            v[2] = splat(c, mkconst(scalar_of(v[0].t), 1), width_of(v[0].t));
            *out = gl(c, G450_FClamp, v, 3, v[0].t);
            return 1;
        }
        *out = gl(c, inst, v, n, v[0].t);
        return 1;
    }
    if (IS("length") || IS("distance")){
        for (int k = 0; k < n; k++) v[k] = loaded(c, call->list[k]);
        common(c, v, n, line);
        *out = gl(c, IS("length") ? G450_Length : G450_Distance, v, n, scalar_of(v[0].t));
        return 1;
    }
    if (IS("cross")){
        v[0] = loaded(c, call->list[0]); v[1] = loaded(c, call->list[1]);
        common(c, v, 2, line);
        *out = gl(c, G450_Cross, v, 2, v[0].t);
        return 1;
    }
    if (IS("refract")){
        v[0] = loaded(c, call->list[0]); v[1] = loaded(c, call->list[1]); v[2] = loaded(c, call->list[2]);
        common(c, v, 2, line);
        v[2] = as_type(c, v[2], scalar_of(v[0].t), line);
        *out = gl(c, G450_Refract, v, 3, v[0].t);
        return 1;
    }
    if (IS("dot")){
        v[0] = loaded(c, call->list[0]); v[1] = loaded(c, call->list[1]);
        common(c, v, 2, line);
        if (!is_float(v[0].t)) wgsl_fail(c->W, line, "dot of integers is not supported");
        uint32_t r = nid(c);
        E(c, OpDot, tid(S, scalar_of(v[0].t)), r, value(c, &v[0]), value(c, &v[1]));
        *out = mkval(scalar_of(v[0].t), r);
        return 1;
    }
    if (IS("determinant")){ v[0] = loaded(c, call->list[0]); *out = gl(c, G450_Determinant, v, 1, t_f32(S)); return 1; }
    if (IS("transpose")){
        v[0] = loaded(c, call->list[0]);
        Type *rt = t_mat(S, v[0].t->e->n, v[0].t->n);
        uint32_t r = nid(c);
        E(c, OpTranspose, tid(S, rt), r, value(c, &v[0]));
        *out = mkval(rt, r);
        return 1;
    }
    if (IS("select")){
        v[0] = loaded(c, call->list[0]); v[1] = loaded(c, call->list[1]); v[2] = loaded(c, call->list[2]);
        common(c, v, 2, line);
        if (!is_bool(v[2].t)) wgsl_fail(c->W, line, "select's condition must be bool");
        if (v[2].has_cv && width_of(v[2].t) == 1){ *out = v[2].cv[0] ? v[1] : v[0]; return 1; }
        if (width_of(v[2].t) != width_of(v[0].t)) v[2] = splat(c, v[2], width_of(v[0].t));
        uint32_t r = nid(c);
        E(c, OpSelect, tid(S, v[0].t), r, value(c, &v[2]), value(c, &v[1]), value(c, &v[0]));
        *out = mkval(v[0].t, r);
        return 1;
    }
    if (IS("any") || IS("all")){
        v[0] = loaded(c, call->list[0]);
        if (width_of(v[0].t) == 1){ *out = v[0]; return 1; }
        uint32_t r = nid(c);
        E(c, IS("any") ? OpAny : OpAll, tid(S, t_bool(S)), r, value(c, &v[0]));
        *out = mkval(t_bool(S), r);
        return 1;
    }
    if (IS("countOneBits") || IS("reverseBits")){
        v[0] = loaded(c, call->list[0]);
        if (is_abstract(v[0].t)) v[0] = as_type(c, v[0], concrete(c, v[0].t), line);
        uint32_t r = nid(c);
        E(c, IS("countOneBits") ? OpBitCount : OpBitReverse, tid(S, v[0].t), r, value(c, &v[0]));
        *out = mkval(v[0].t, r);
        return 1;
    }
    if (IS("countLeadingZeros") || IS("countTrailingZeros") || IS("firstLeadingBit") || IS("firstTrailingBit")){
        v[0] = loaded(c, call->list[0]);
        if (is_abstract(v[0].t)) v[0] = as_type(c, v[0], concrete(c, v[0].t), line);
        Type *t = v[0].t, *ut = t_vec(S, t_u32(S), width_of(t));
        Val x = convert(c, v[0], ut, line);              // (the bits as u32)
        Val r;
        if (IS("firstLeadingBit")){ Val a[1] = {is_sint(t) ? v[0] : x}; r = gl(c, is_sint(t) ? G450_FindSMsb : G450_FindUMsb, a, 1, is_sint(t) ? t : ut); r = convert(c, r, t, line); *out = r; return 1; }
        if (IS("firstTrailingBit")){ Val a[1] = {x}; r = gl(c, G450_FindILsb, a, 1, ut); *out = convert(c, r, t, line); return 1; }
        if (IS("countLeadingZeros")){
            Val a[1] = {x};
            Val msb = gl(c, G450_FindUMsb, a, 1, ut);    // -1 for 0: 31 - (-1) = 32
            r = binop(c, TK_MINUS, splat(c, mkconst(t_u32(S), 31), width_of(t)), msb, line);
        } else {
            Val a[1] = {x};
            Val lsb = gl(c, G450_FindILsb, a, 1, ut);    // -1 (all ones) for 0: min with 32
            Val b[2] = {lsb, splat(c, mkconst(t_u32(S), 32), width_of(t))};
            r = gl(c, G450_UMin, b, 2, ut);
        }
        *out = convert(c, r, t, line);
        return 1;
    }
    if (IS("pack2x16float") || IS("pack2x16snorm") || IS("pack2x16unorm") || IS("pack4x8snorm") || IS("pack4x8unorm")){
        v[0] = loaded(c, call->list[0]);
        if (is_abstract(v[0].t)) v[0] = as_type(c, v[0], concrete(c, v[0].t), line);
        uint32_t inst = IS("pack2x16float") ? G450_PackHalf2x16 : IS("pack2x16snorm") ? G450_PackSnorm2x16 : IS("pack2x16unorm") ? G450_PackUnorm2x16 :
                        IS("pack4x8snorm") ? G450_PackSnorm4x8 : G450_PackUnorm4x8;
        *out = gl(c, inst, v, 1, t_u32(S));
        return 1;
    }
    if (IS("unpack2x16float") || IS("unpack2x16snorm") || IS("unpack2x16unorm") || IS("unpack4x8snorm") || IS("unpack4x8unorm")){
        v[0] = loaded(c, call->list[0]);
        v[0] = convert(c, v[0], t_u32(S), line);
        int four = s[6] == '4';
        uint32_t inst = IS("unpack2x16float") ? G450_UnpackHalf2x16 : IS("unpack2x16snorm") ? G450_UnpackSnorm2x16 : IS("unpack2x16unorm") ? G450_UnpackUnorm2x16 :
                        IS("unpack4x8snorm") ? G450_UnpackSnorm4x8 : G450_UnpackUnorm4x8;
        *out = gl(c, inst, v, 1, t_vec(S, t_f32(S), four ? 4 : 2));
        return 1;
    }
    if (IS("dpdx") || IS("dpdy") || IS("fwidth") || IS("dpdxFine") || IS("dpdyFine") || IS("dpdxCoarse") || IS("dpdyCoarse")){
        v[0] = loaded(c, call->list[0]);
        if (is_abstract(v[0].t)) v[0] = as_type(c, v[0], concrete(c, v[0].t), line);
        uint32_t o = IS("dpdx") ? OpDPdx : IS("dpdy") ? OpDPdy : IS("fwidth") ? OpFwidth : IS("dpdxFine") ? OpDPdxFine :
                     IS("dpdyFine") ? OpDPdyFine : IS("dpdxCoarse") ? OpDPdxCoarse : OpDPdyCoarse;
        if (l > 6) S->cap_derivative = 1;
        uint32_t r = nid(c);
        E(c, o, tid(S, v[0].t), r, value(c, &v[0]));
        *out = mkval(v[0].t, r);
        return 1;
    }
    if (IS("bitcast")){
        Type *to = resolve_type(c, call->type->list[0]);
        v[0] = loaded(c, call->list[0]);
        if (is_abstract(v[0].t)) v[0] = as_type(c, v[0], concrete(c, v[0].t), line);
        if (v[0].t == to){ *out = v[0]; return 1; }
        if (width_of(v[0].t) != width_of(to)) wgsl_fail(c->W, line, "bitcast between different sizes");
        uint32_t r = nid(c);
        E(c, OpBitcast, tid(S, to), r, value(c, &v[0]));
        *out = mkval(to, r);
        return 1;
    }
    if (IS("workgroupBarrier") || IS("storageBarrier")){
        // execution and memory: workgroup scope; storage: device memory made visible too
        int st = IS("storageBarrier");
        E(c, OpControlBarrier, c_u32(S, 2), c_u32(S, st ? 1 : 2), c_u32(S, st ? 0x48 : 0x108));
        (void)barrier_scope_sem;
        *out = mkval(t_void(S), 0);
        return 1;
    }
    if (IS("workgroupUniformLoad")){
        Val p = ptr_arg(c, call->list[0], line);
        E(c, OpControlBarrier, c_u32(S, 2), c_u32(S, 2), c_u32(S, 0x108));
        Val r; memset(&r, 0, sizeof(r)); r.t = p.t->e; r.id = p.id; r.ref = 1; r.sc = p.t->sc;
        if (r.t->k == T_ATOMIC){
            uint32_t x = nid(c);
            E(c, OpAtomicLoad, tid(S, r.t->e), x, r.id, c_u32(S, 2), c_u32(S, 0));
            r = mkval(r.t->e, x);
        } else load(c, &r);
        E(c, OpControlBarrier, c_u32(S, 2), c_u32(S, 2), c_u32(S, 0x108));
        *out = r;
        return 1;
    }
    if (l > 6 && !memcmp(s, "atomic", 6)){
        Val p = ptr_arg(c, call->list[0], line);
        if (p.t->e->k != T_ATOMIC) wgsl_fail(c->W, line, "%.*s needs a pointer to an atomic", l, s);
        Type *t = p.t->e->e;
        uint32_t scope = c_u32(S, p.t->sc == SC_Workgroup ? 2 : 1), sem = c_u32(S, 0), r = nid(c);
        if (IS("atomicLoad")){ E(c, OpAtomicLoad, tid(S, t), r, p.id, scope, sem); *out = mkval(t, r); return 1; }
        Val x = n > 1 ? loaded(c, call->list[1]) : mkconst(t, 0);
        if (is_abstract(x.t)) x = as_type(c, x, t, line);
        if (IS("atomicStore")){ E(c, OpAtomicStore, p.id, scope, sem, value(c, &x)); *out = mkval(t_void(S), 0); return 1; }
        if (IS("atomicCompareExchangeWeak")){
            Val y = loaded(c, call->list[2]);
            if (is_abstract(y.t)) y = as_type(c, y, t, line);
            uint32_t cmp = value(c, &x), val = value(c, &y);
            E(c, OpAtomicCompareExchange, tid(S, t), r, p.id, scope, sem, sem, val, cmp);
            // the result: struct { old_value, exchanged }
            static StructInfo si_u, si_i;
            StructInfo *si = t->k == T_U32 ? &si_u : &si_i;
            static const char *names[2] = {"old_value", "exchanged"};
            static int lens[2] = {9, 9};
            static uint32_t offs[2] = {0, 4};
            si->name = t->k == T_U32 ? "__atomic_compare_exchange_result_u32" : "__atomic_compare_exchange_result_i32";
            si->nlen = (int)strlen(si->name);
            si->n = 2; si->mname = names; si->mlen = lens; si->off = offs; si->size = 8; si->align = 4;
            Type *mts[2] = {t, t_bool(S)};
            si->mt = wgsl_alloc(c->W, sizeof(mts));
            memcpy(si->mt, mts, sizeof(mts));
            Type *rt = mk(S, T_STRUCT, NULL, 0, 0, si);
            uint32_t eq = nid(c), res = nid(c);
            E(c, OpIEqual, tid(S, t_bool(S)), eq, r, cmp);
            E(c, OpCompositeConstruct, tid(S, rt), res, r, eq);
            *out = mkval(rt, res);
            return 1;
        }
        uint32_t o = IS("atomicAdd") ? OpAtomicIAdd : IS("atomicSub") ? OpAtomicISub : IS("atomicMax") ? (t->k == T_U32 ? OpAtomicUMax : OpAtomicSMax) :
                     IS("atomicMin") ? (t->k == T_U32 ? OpAtomicUMin : OpAtomicSMin) : IS("atomicAnd") ? OpAtomicAnd : IS("atomicOr") ? OpAtomicOr :
                     IS("atomicXor") ? OpAtomicXor : IS("atomicExchange") ? OpAtomicExchange : 0;
        if (!o) wgsl_fail(c->W, line, "unknown %.*s", l, s);
        E(c, o, tid(S, t), r, p.id, scope, sem, value(c, &x));
        *out = mkval(t, r);
        return 1;
    }
    if (IS("arrayLength")){
        Node *a = call->list[0];
        if (a->k != N_ADDR) wgsl_fail(c->W, line, "arrayLength(&buffer)");
        Node *x = a->a;
        int member = 0;
        Node *g = x;
        if (x->k == N_MEMBER){ g = x->a; }
        Sym *sym = g->k == N_IDENT ? lookup(c, g->name, g->nlen) : NULL;
        if (!sym || sym->kind != SYM_GLOBAL) wgsl_fail(c->W, line, "arrayLength of a storage buffer's runtime array");
        resolve_global(c, sym->index);
        GlobalInfo *gi = &c->gl[sym->index];
        if (x->k == N_MEMBER){
            if (gi->wrapped || gi->t->k != T_STRUCT) wgsl_fail(c->W, line, "arrayLength: not a runtime array");
            StructInfo *si = gi->t->s;
            member = si->n - 1;
        } else if (!gi->wrapped) wgsl_fail(c->W, line, "arrayLength: not a runtime array");
        if (c->fn) c->fn->uses[sym->index] = 1;
        uint32_t r = nid(c);
        E(c, OpArrayLength, tid(S, t_u32(S)), r, gi->var, (uint32_t)member);
        *out = mkval(t_u32(S), r);
        return 1;
    }
    if (IS("textureLoad") || IS("textureDimensions")){
        Val tex = eval(c, call->list[0]);
        if (!tex.ref || (tex.t->k != T_TEX && tex.t->k != T_TEXDEPTH)) wgsl_fail(c->W, line, "%.*s on a texture", l, s);
        int depth = tex.t->k == T_TEXDEPTH;
        uint32_t img = load(c, &tex);
        if (IS("textureDimensions")){
            S->cap_image_query = 1;
            Val lod = n > 1 ? convert(c, loaded(c, call->list[1]), t_i32(S), line) : mkconst(t_i32(S), 0);
            Type *v2i = t_vec(S, t_i32(S), 2), *v2u = t_vec(S, t_u32(S), 2);
            uint32_t r = nid(c), u = nid(c);
            E(c, OpImageQuerySizeLod, tid(S, v2i), r, img, value(c, &lod));
            E(c, OpBitcast, tid(S, v2u), u, r);
            *out = mkval(v2u, u);
            return 1;
        }
        Val co = loaded(c, call->list[1]);
        if (is_abstract(co.t)) co = as_type(c, co, t_vec(S, t_i32(S), 2), line);
        co = convert(c, co, t_vec(S, t_i32(S), 2), line);
        Val lod = convert(c, loaded(c, call->list[2]), t_i32(S), line);
        Type *v4 = t_vec(S, t_f32(S), 4);
        uint32_t r = nid(c);
        E(c, OpImageFetch, tid(S, v4), r, img, value(c, &co), 0x2, value(c, &lod));
        if (depth){ uint32_t x = nid(c); E(c, OpCompositeExtract, tid(S, t_f32(S)), x, r, 0); *out = mkval(t_f32(S), x); }
        else *out = mkval(v4, r);
        return 1;
    }
#undef IS
    return 0;
}

// ---- expressions ----
static void resolve_fn(Cx *c, int i);

static Val call_fn(Cx *c, Node *call, Sym *sym){
    Sp *S = SP(c);
    resolve_fn(c, sym->index);
    FnInfo *f = &c->fns[sym->index];
    if (f->d->stage) wgsl_fail(c->W, call->line, "an entry point cannot be called");
    if (c->fn) c->fn->calls[sym->index] = 1;
    if (call->n != f->d->np) wgsl_fail(c->W, call->line, "%.*s takes %d arguments", f->d->nlen, f->d->name, f->d->np);
    uint32_t *a = wgsl_alloc(c->W, (size_t)(3 + call->n) * sizeof(uint32_t));
    a[0] = tid(S, f->ret); a[1] = nid(c); a[2] = f->id;
    for (int i = 0; i < call->n; i++){
        Type *pt = f->pt[i];
        Val v;
        if (pt->k == T_PTR){
            v = ptr_arg(c, call->list[i], call->line);
            if (v.t->e != pt->e) wgsl_fail(c->W, call->line, "argument %d: a pointer to the wrong type", i + 1);
        } else {
            v = loaded(c, call->list[i]);
            if (is_abstract(v.t)) v = as_type(c, v, pt, call->line);
            if (v.t != pt){ char p[64], q[64]; type_name(v.t, p, sizeof(p)); type_name(pt, q, sizeof(q)); wgsl_fail(c->W, call->line, "argument %d: expected %s, got %s", i + 1, q, p); }
        }
        a[3 + i] = value(c, &v);
    }
    op(S, FB(c), OpFunctionCall, 3 + call->n, a);
    return mkval(f->ret, a[1]);
}

// short-circuit && and ||: the right side only when it matters
static Val logic(Cx *c, Node *e){
    Sp *S = SP(c);
    Val a = loaded(c, e->a);
    if (!is_bool(a.t) || width_of(a.t) != 1) wgsl_fail(c->W, e->line, "&& and || take bool");
    if (a.has_cv){
        if ((e->op == TK_ANDAND) != (a.cv[0] != 0)) return mkconst(t_bool(S), a.cv[0] != 0);
        Val b = loaded(c, e->b);
        return b;
    }
    uint32_t x = value(c, &a), from = c->block, rhs = nid(c), merge = nid(c);
    E(c, OpSelectionMerge, merge, 0);
    if (e->op == TK_ANDAND) E(c, OpBranchConditional, x, rhs, merge);
    else E(c, OpBranchConditional, x, merge, rhs);
    label(c, rhs);
    Val b = loaded(c, e->b);
    if (!is_bool(b.t) || width_of(b.t) != 1) wgsl_fail(c->W, e->line, "&& and || take bool");
    uint32_t y = value(c, &b), rhs_end = c->block;
    branch(c, merge);
    label(c, merge);
    uint32_t r = nid(c);
    E(c, OpPhi, tid(S, t_bool(S)), r, x, from, y, rhs_end);
    return mkval(t_bool(S), r);
}

static Type *type_by_name(Cx *c, const char *s, int l){
    // a type named in a call (a constructor or a conversion), or NULL
    static const char *names[] = {"bool", "i32", "u32", "f32", "vec2f", "vec3f", "vec4f", "vec2u", "vec3u", "vec4u",
                                  "vec2i", "vec3i", "vec4i", "mat2x2f", "mat2x3f", "mat2x4f", "mat3x2f", "mat3x3f",
                                  "mat3x4f", "mat4x2f", "mat4x3f", "mat4x4f"};
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++)
        if (wgsl_name_is(s, l, names[i])){ Node t; memset(&t, 0, sizeof(t)); t.k = N_TYPE; t.name = s; t.nlen = l; return resolve_type(c, &t); }
    Sym *sym = lookup(c, s, l);
    if (sym && sym->kind == SYM_STRUCT) return resolve_struct(c, sym->index);
    return NULL;
}

static Val eval(Cx *c, Node *e){
    Sp *S = SP(c);
    switch (e->k){
    case N_INT: {
        Type *t = e->suffix == 'u' ? t_u32(S) : e->suffix == 'i' ? t_i32(S) : t_aint(S);
        if (e->suffix == 'u' && e->ival > 0xffffffffull) wgsl_fail(c->W, e->line, "the number is too large for u32");
        return mkconst(t, (double)e->ival);
    }
    case N_FLOAT: return mkconst(e->suffix == 'f' ? t_f32(S) : t_afloat(S), e->suffix == 'f' ? (double)(float)e->fval : e->fval);
    case N_BOOL: return mkconst(t_bool(S), (double)e->ival);
    case N_TYPE:          // (a name in a template's arguments: a constant)
        if (e->n) wgsl_fail(c->W, e->line, "a type where a value was expected");
        /* fallthrough */
    case N_IDENT: {
        Sym *sym = lookup(c, e->name, e->nlen);
        if (!sym) wgsl_fail(c->W, e->line, "unknown name '%.*s'", e->nlen, e->name);
        if (sym->kind == SYM_GLOBAL){
            resolve_global(c, sym->index);
            GlobalInfo *g = &c->gl[sym->index];
            if (c->W->globals[sym->index].is_const) return g->cv;
            if (c->fn) c->fn->uses[sym->index] = 1;
            Val v; memset(&v, 0, sizeof(v));
            v.t = g->t; v.ref = 1; v.sc = g->sc;
            if (g->wrapped){
                uint32_t r = nid(c);
                E(c, OpAccessChain, tid(S, t_ptr(S, g->t, g->sc)), r, g->var, c_u32(S, 0));
                v.id = r;
            } else v.id = g->var;
            return v;
        }
        if (sym->kind == SYM_FN || sym->kind == SYM_STRUCT) wgsl_fail(c->W, e->line, "'%.*s' is not a value", e->nlen, e->name);
        return sym->v;
    }
    case N_UNARY: return unop(c, e->op, eval(c, e->a), e->line);
    case N_BINARY:
        if (e->op == TK_ANDAND || e->op == TK_OROR) return logic(c, e);
        return binop(c, e->op, eval(c, e->a), eval(c, e->b), e->line);
    case N_MEMBER: return member(c, eval(c, e->a), e->name, e->nlen, e->line);
    case N_INDEX: { Val a = eval(c, e->a); Val b = eval(c, e->b); return index_of(c, a, b, e->line); }
    case N_ADDR: return ptr_arg(c, e->a, e->line);
    case N_DEREF: {
        Val p = loaded(c, e->a);
        if (p.t->k != T_PTR) wgsl_fail(c->W, e->line, "* of something that is not a pointer");
        return as_ref(p);
    }
    case N_CALL: {
        Val out;
        if (e->type && !wgsl_name_is(e->name, e->nlen, "bitcast")) return construct(c, resolve_type(c, e->type), e);
        Sym *sym = lookup(c, e->name, e->nlen);
        if (sym && sym->kind == SYM_FN) return call_fn(c, e, sym);
        Type *t = type_by_name(c, e->name, e->nlen);
        if (t) return construct(c, t, e);
        if (builtin(c, e, &out)) return out;
        wgsl_fail(c->W, e->line, "unknown function '%.*s'", e->nlen, e->name);
    }
    /* fallthrough */
    default:
        wgsl_fail(c->W, e->line, "not an expression");
    }
    return mkconst(t_bool(S), 0);
}

// ---- statements ----
static void stmt(Cx *c, Node *s);
static void ret_entry(Cx *c, Val *v, int line);

static void store(Cx *c, Val dst, Val v, int line){
    if (!dst.ref) wgsl_fail(c->W, line, "cannot assign to this");
    if (dst.t->k == T_ATOMIC) wgsl_fail(c->W, line, "an atomic must be written with atomicStore");
    if (v.ref) load(c, &v);
    if (is_abstract(v.t)) v = as_type(c, v, dst.t, line);
    if (v.t != dst.t){ char p[64], q[64]; type_name(v.t, p, sizeof(p)); type_name(dst.t, q, sizeof(q)); wgsl_fail(c->W, line, "cannot assign %s to %s", p, q); }
    uint32_t x = value(c, &v);
    if (buffer_sc(dst.sc) && layout_differs(dst.t)){
        uint32_t y = nid(c);
        E(c, OpCopyLogical, tid_l(SP(c), dst.t), y, x);
        x = y;
    }
    E(c, OpStore, dst.id, x);
}

static void block_stmts(Cx *c, Node *b){
    scope_open(c);
    for (int i = 0; i < b->n && !c->dead; i++) stmt(c, b->list[i]);
    scope_close(c);
}

static void declare(Cx *c, const char *name, int nlen, int kind, Val v){
    Sym s; memset(&s, 0, sizeof(s));
    s.name = name; s.nlen = nlen; s.kind = kind; s.v = v;
    sym_push(c, s);
}

static int assign_op(int op){
    switch (op){
    case TK_PLUSEQ: return TK_PLUS; case TK_MINUSEQ: return TK_MINUS; case TK_STAREQ: return TK_STAR;
    case TK_SLASHEQ: return TK_SLASH; case TK_PERCENTEQ: return TK_PERCENT; case TK_AMPEQ: return TK_AMP;
    case TK_PIPEEQ: return TK_PIPE; case TK_CARETEQ: return TK_CARET; case TK_SHLEQ: return TK_SHL; case TK_SHREQ: return TK_SHR;
    }
    return 0;
}

static void loop(Cx *c, Node *init, Node *cond, Node *cont, Node *body){
    scope_open(c);
    if (init) stmt(c, init);
    uint32_t header = nid(c), check = nid(c), bodyl = nid(c), contl = nid(c), merge = nid(c);
    branch(c, header);
    label(c, header);
    E(c, OpLoopMerge, merge, contl, 0);
    E(c, OpBranch, check);
    label(c, check);
    if (cond){
        Val v = loaded(c, cond);
        if (!is_bool(v.t) || width_of(v.t) != 1) wgsl_fail(c->W, cond->line, "a loop's condition must be bool");
        E(c, OpBranchConditional, value(c, &v), bodyl, merge);
        c->dead = 1;
    } else branch(c, bodyl);
    label(c, bodyl);
    if (c->nloops >= 32) wgsl_fail(c->W, body->line, "loops nested too deep");
    c->loops[c->nloops].merge = merge; c->loops[c->nloops].cont = contl; c->nloops++;
    block_stmts(c, body);
    c->nloops--;
    branch(c, contl);
    label(c, contl);
    if (cont) stmt(c, cont);
    branch(c, header);
    label(c, merge);
    scope_close(c);
}

static void stmt(Cx *c, Node *s){
    Sp *S = SP(c);
    if (c->dead) return;
    switch (s->k){
    case N_BLOCK: block_stmts(c, s); return;
    case N_LET: case N_CONST: {
        Val v = loaded(c, s->a);
        if (s->type){
            Type *t = resolve_type(c, s->type);
            if (is_abstract(v.t)) v = as_type(c, v, t, s->line);
            if (v.t != t){ char p[64], q[64]; type_name(v.t, p, sizeof(p)); type_name(t, q, sizeof(q)); wgsl_fail(c->W, s->line, "expected %s, got %s", q, p); }
        } else if (is_abstract(v.t) && s->k == N_LET) v = as_type(c, v, concrete(c, v.t), s->line);
        if (s->k == N_CONST && !v.has_cv) wgsl_fail(c->W, s->line, "a const needs a constant value");
        declare(c, s->name, s->nlen, SYM_VALUE, v);
        return;
    }
    case N_VAR: {
        Type *t = NULL;
        Val v;
        memset(&v, 0, sizeof(v));
        if (s->a){ v = loaded(c, s->a); }
        if (s->type) t = resolve_type(c, s->type);
        else if (s->a) t = concrete(c, v.t);
        else wgsl_fail(c->W, s->line, "a var needs a type or a value");
        uint32_t var = fvar(c, t, 0);
        Val ref; memset(&ref, 0, sizeof(ref)); ref.t = t; ref.id = var; ref.ref = 1; ref.sc = SC_Function;
        // (stored here, not as the variable's initializer: a declaration in a loop starts again each time)
        if (s->a) store(c, ref, v, s->line);
        else E(c, OpStore, var, (is_scalar(t) || t->k == T_VEC) ? const_id(c, t, mkconst(t, 0).cv) : c_null(S, t));
        declare(c, s->name, s->nlen, SYM_VAR, ref);
        return;
    }
    case N_ASSIGN: {
        Val dst = eval(c, s->a);
        dst = as_ref(dst);
        Val v = eval(c, s->b);
        if (s->op != TK_EQ){
            Val cur = dst;
            load(c, &cur);
            v = binop(c, assign_op(s->op), cur, v, s->line);
        }
        store(c, dst, v, s->line);
        return;
    }
    case N_INCDEC: {
        Val dst = as_ref(eval(c, s->a));
        Val cur = dst;
        load(c, &cur);
        Val v = binop(c, s->op == TK_PLUSPLUS ? TK_PLUS : TK_MINUS, cur, mkconst(t_aint(S), 1), s->line);
        store(c, dst, v, s->line);
        return;
    }
    case N_IF: {
        Val v = loaded(c, s->a);
        if (!is_bool(v.t) || width_of(v.t) != 1) wgsl_fail(c->W, s->line, "an if's condition must be bool");
        uint32_t cv = value(c, &v), then = nid(c), merge = nid(c), els = s->c ? nid(c) : merge;
        E(c, OpSelectionMerge, merge, 0);
        E(c, OpBranchConditional, cv, then, els);
        c->dead = 1;
        label(c, then);
        if (s->b->k == N_BLOCK) block_stmts(c, s->b); else stmt(c, s->b);
        branch(c, merge);
        if (s->c){
            label(c, els);
            if (s->c->k == N_BLOCK) block_stmts(c, s->c); else stmt(c, s->c);
            branch(c, merge);
        }
        label(c, merge);
        return;
    }
    case N_FOR: loop(c, s->a, s->b, s->c, s->d); return;
    case N_WHILE: loop(c, NULL, s->a, NULL, s->b); return;
    case N_LOOP: loop(c, NULL, NULL, NULL, s->a); return;
    case N_BREAK:
        if (!c->nloops) wgsl_fail(c->W, s->line, "break outside a loop");
        branch(c, c->loops[c->nloops - 1].merge);
        return;
    case N_CONTINUE:
        if (!c->nloops) wgsl_fail(c->W, s->line, "continue outside a loop");
        branch(c, c->loops[c->nloops - 1].cont);
        return;
    case N_BREAKIF: wgsl_fail(c->W, s->line, "break if is not supported"); return;
    case N_DISCARD: E(c, OpKill); c->dead = 1; return;
    case N_RETURN: {
        if (c->stage){
            Val v;
            if (s->a) v = loaded(c, s->a);
            ret_entry(c, s->a ? &v : NULL, s->line);
            E(c, OpReturn);
            c->dead = 1;
            return;
        }
        Type *rt = c->fn->ret;
        if (!s->a){
            if (rt->k != T_VOID) wgsl_fail(c->W, s->line, "return needs a value");
            E(c, OpReturn);
        } else {
            Val v = loaded(c, s->a);
            if (is_abstract(v.t)) v = as_type(c, v, rt, s->line);
            if (v.t != rt){ char p[64], q[64]; type_name(v.t, p, sizeof(p)); type_name(rt, q, sizeof(q)); wgsl_fail(c->W, s->line, "returns %s, expected %s", p, q); }
            E(c, OpReturnValue, value(c, &v));
        }
        c->dead = 1;
        return;
    }
    case N_EXPR: { Val v = eval(c, s->a); (void)v; return; }
    default: wgsl_fail(c->W, s->line, "not a statement");
    }
}

// ---- module-scope variables and constants ----
static void name_it(Cx *c, uint32_t id, const char *s, int n){ op_str(SP(c), &SP(c)->names, OpName, 1, &id, s, n, 0, NULL); }

static void resolve_global(Cx *c, int i){
    Sp *S = SP(c);
    GlobalInfo *g = &c->gl[i];
    GlobalDecl *d = &c->W->globals[i];
    if (g->resolved == 1) return;
    if (g->resolved == 2) wgsl_fail(c->W, d->line, "'%.*s' depends on itself", d->nlen, d->name);
    g->resolved = 2;
    if (d->is_const){
        // (evaluated with no function around: only constants can come out)
        Words saved = S->fbody;
        memset(&S->fbody, 0, sizeof(S->fbody));
        Val v = eval(c, d->init);
        if (d->type){
            Type *t = resolve_type(c, d->type);
            if (is_abstract(v.t)) v = as_type(c, v, t, d->line);
            if (v.t != t) wgsl_fail(c->W, d->line, "the const's value has another type");
        }
        if (!v.has_cv || S->fbody.n) wgsl_fail(c->W, d->line, "a const needs a constant value");
        free(S->fbody.w);
        S->fbody = saved;
        g->cv = v;
        g->resolved = 1;
        return;
    }
    Type *t = resolve_type(c, d->type);
    g->t = t;
    int sc = d->space == SP_PRIVATE ? SC_Private : d->space == SP_WORKGROUP ? SC_Workgroup : d->space == SP_UNIFORM ? SC_Uniform :
             d->space == SP_STORAGE ? SC_StorageBuffer : SC_UniformConstant;
    g->sc = sc;
    uint32_t var = new_id(S);
    g->var = var;
    if (sc == SC_Uniform || sc == SC_StorageBuffer){
        int ends_rt = t->k == T_STRUCT && t->s->n && t->s->mt[t->s->n - 1]->k == T_RTARRAY;
        Type *vt;
        if (ends_rt){
            vt = t;
            g->wrapped = 0;
        } else {
            // the variable as a struct with one member: what Vulkan's buffer interface wants
            StructInfo *si = wgsl_alloc(c->W, sizeof(StructInfo));
            si->name = d->name; si->nlen = d->nlen;
            si->n = 1;
            si->mname = wgsl_alloc(c->W, sizeof(char *)); si->mname[0] = "value";
            si->mlen = wgsl_alloc(c->W, sizeof(int)); si->mlen[0] = 5;
            si->mt = wgsl_alloc(c->W, sizeof(Type *)); si->mt[0] = t;
            si->off = wgsl_alloc(c->W, sizeof(uint32_t));
            si->align = align_of(t); si->size = size_of(t);
            vt = mk(S, T_STRUCT, NULL, 0, 0, si);
            g->wrapped = 1;
        }
        uint32_t vid = tid_l(S, vt);
        OP(S, &S->decos, OpDecorate, vid, DecBlock);
        if (sc == SC_StorageBuffer && d->access == AC_READ)
            for (int k = 0; k < vt->s->n; k++) OP(S, &S->decos, OpMemberDecorate, vid, (uint32_t)k, DecNonWritable);
        OP(S, &S->globals, OpVariable, tid(S, t_ptr(S, vt, sc)), var, (uint32_t)sc);
    } else if (sc == SC_UniformConstant){
        OP(S, &S->globals, OpVariable, tid(S, t_ptr(S, t, sc)), var, (uint32_t)sc);
    } else {
        // private and workgroup memory start at zero (WGSL); a private one may start at its constant
        uint32_t init;
        if (d->init){
            Words saved = S->fbody;
            memset(&S->fbody, 0, sizeof(S->fbody));
            Val v = eval(c, d->init);
            if (is_abstract(v.t)) v = as_type(c, v, t, d->line);
            if (!v.has_cv || S->fbody.n) wgsl_fail(c->W, d->line, "a module-scope var starts at a constant");
            free(S->fbody.w);
            S->fbody = saved;
            init = const_id(c, t, v.cv);
        } else init = (sc != SC_Workgroup && (is_scalar(t) || t->k == T_VEC)) ? const_id(c, t, mkconst(t, 0).cv) : c_null(S, t);
        OP(S, &S->globals, OpVariable, tid(S, t_ptr(S, t, sc)), var, (uint32_t)sc, init);
    }
    if (d->group >= 0) OP(S, &S->decos, OpDecorate, var, DecDescriptorSet, (uint32_t)d->group);
    if (d->binding >= 0) OP(S, &S->decos, OpDecorate, var, DecBinding, (uint32_t)d->binding);
    name_it(c, var, d->name, d->nlen);
    g->resolved = 1;
}

// ---- functions ----
static uint32_t fn_type(Cx *c, Type *ret, Type **pt, int n){
    Sp *S = SP(c);
    // (function types once each: SPIR-V does not allow two the same)
    for (int i = 0; i < c->nfnty; i++)
        if (c->fnty[i].ret == ret && c->fnty[i].n == n && (!n || !memcmp(c->fnty[i].pt, pt, (size_t)n * sizeof(Type *)))) return c->fnty[i].id;
    if (n > 16) wgsl_fail(c->W, 0, "too many parameters");
    uint32_t r = tid(S, ret);
    uint32_t a[18];
    for (int i = 0; i < n; i++) a[2 + i] = tid(S, pt[i]);
    a[0] = new_id(S); a[1] = r;
    op(S, &S->globals, OpTypeFunction, 2 + n, a);
    if (c->nfnty < 256){ c->fnty[c->nfnty].ret = ret; if (n) memcpy(c->fnty[c->nfnty].pt, pt, (size_t)n * sizeof(Type *)); c->fnty[c->nfnty].n = n; c->fnty[c->nfnty].id = a[0]; c->nfnty++; }
    return a[0];
}

static void resolve_fn(Cx *c, int i){
    FnInfo *f = &c->fns[i];
    if (f->resolved) return;
    FnDecl *d = f->d;
    f->pt = wgsl_alloc(c->W, (size_t)(d->np ? d->np : 1) * sizeof(Type *));
    for (int k = 0; k < d->np; k++) f->pt[k] = resolve_type(c, d->p[k].type);
    f->ret = d->ret ? resolve_type(c, d->ret) : t_void(SP(c));
    f->id = new_id(SP(c));
    f->resolved = 1;
}

static void iface_add(Cx *c, uint32_t var){
    if (c->niface == c->cap_iface){
        int cap = c->cap_iface ? 2 * c->cap_iface : 16;
        uint32_t *v = realloc(c->iface, (size_t)cap * sizeof(uint32_t));
        if (!v) wgsl_fail(c->W, 0, "out of memory");
        c->iface = v; c->cap_iface = cap;
    }
    c->iface[c->niface++] = var;
}

static uint32_t bi_code(int b, int output, int stage){
    switch (b){
    case BI_GLOBAL_ID: return 28; case BI_LOCAL_ID: return 27; case BI_LOCAL_INDEX: return 29; case BI_WORKGROUP_ID: return 26;
    case BI_NUM_WORKGROUPS: return 24; case BI_VERTEX_INDEX: return 42; case BI_INSTANCE_INDEX: return 43;
    case BI_POSITION: return (output || stage == ST_VERTEX) ? 0 : 15;
    case BI_FRONT_FACING: return 17; case BI_FRAG_DEPTH: return 22;
    }
    return 0;
}

// an interface variable (input or output) for one value with its attributes
static uint32_t io_var(Cx *c, Type *t, Attrs at, int output, int line){
    Sp *S = SP(c);
    if (!at.builtin && at.location < 0) wgsl_fail(c->W, line, "an entry point's input or output needs @builtin or @location");
    uint32_t v = new_id(S);
    OP(S, &S->globals, OpVariable, tid(S, t_ptr(S, t, output ? SC_Output : SC_Input)), v, output ? SC_Output : SC_Input);
    if (at.builtin) OP(S, &S->decos, OpDecorate, v, DecBuiltIn, bi_code(at.builtin, output, c->stage));
    else OP(S, &S->decos, OpDecorate, v, DecLocation, (uint32_t)at.location);
    if (!at.builtin && (at.flat || is_int(t)) && ((c->stage == ST_FRAGMENT && !output) || (c->stage == ST_VERTEX && output)))
        OP(S, &S->decos, OpDecorate, v, DecFlat);
    if (at.builtin == BI_FRAG_DEPTH) c->uses_frag_depth = 1;
    iface_add(c, v);
    return v;
}

// an entry point's return value into its outputs (the position's y turned: WebGPU's clip
// space has y up, Vulkan's down)
static void ret_entry(Cx *c, Val *v, int line){
    Sp *S = SP(c);
    if (!v){ if (c->nouts) wgsl_fail(c->W, line, "return needs a value"); return; }
    if (is_abstract(v->t)) *v = as_type(c, *v, concrete(c, v->t), line);
    for (int i = 0; i < c->nouts; i++){
        Val x = *v;
        if (c->out_is_struct){
            uint32_t r = nid(c);
            E(c, OpCompositeExtract, tid(S, c->outs[i].t), r, value(c, v), (uint32_t)i);
            x = mkval(c->outs[i].t, r);
        }
        uint32_t xv = value(c, &x);
        if (c->stage == ST_VERTEX && c->outs[i].t == t_vec(S, t_f32(S), 4) && c->outs[i].var == (uint32_t)-1){}
        E(c, OpStore, c->outs[i].var, xv);
    }
}

static void emit_fn(Cx *c, int i){
    Sp *S = SP(c);
    resolve_fn(c, i);
    FnInfo *f = &c->fns[i];
    FnDecl *d = f->d;
    c->fn = f;
    c->stage = d->stage;
    c->nloops = 0;
    c->nouts = 0;
    c->out_is_struct = 0;
    c->niface = 0;
    c->uses_frag_depth = 0;
    Words *fw = &S->fns;
    uint32_t ftype = d->stage ? fn_type(c, t_void(S), NULL, 0) : fn_type(c, f->ret, f->pt, d->np);
    OP(S, fw, OpFunction, tid(S, d->stage ? t_void(S) : f->ret), f->id, 0, ftype);
    name_it(c, f->id, d->name, d->nlen);
    scope_open(c);
    uint32_t *pid = wgsl_alloc(c->W, (size_t)(d->np ? d->np : 1) * sizeof(uint32_t));
    if (!d->stage) for (int k = 0; k < d->np; k++){ pid[k] = new_id(S); OP(S, fw, OpFunctionParameter, tid(S, f->pt[k]), pid[k]); }
    uint32_t first = new_id(S);
    OP(S, fw, OpLabel, first);
    c->block = first;
    c->dead = 0;
    memset(&S->fvars, 0, sizeof(S->fvars));
    memset(&S->fbody, 0, sizeof(S->fbody));
    // the entry point's position output (vertex), kept to turn its y
    uint32_t pos_out = 0;
    if (d->stage){
        // inputs: each parameter (a struct: each member) from its own variable
        for (int k = 0; k < d->np; k++){
            Type *t = f->pt[k];
            uint32_t val;
            if (t->k == T_STRUCT){
                StructInfo *si = t->s;
                uint32_t *m = wgsl_alloc(c->W, (size_t)(si->n + 2) * sizeof(uint32_t));
                m[0] = tid(S, t); m[1] = nid(c);
                for (int j = 0; j < si->n; j++){
                    uint32_t var = io_var(c, si->mt[j], si->mat[j], 0, d->line), x = nid(c);
                    E(c, OpLoad, tid(S, si->mt[j]), x, var);
                    m[2 + j] = x;
                }
                op(S, FB(c), OpCompositeConstruct, 2 + si->n, m);
                val = m[1];
            } else {
                uint32_t var = io_var(c, t, d->p[k].at, 0, d->line);
                val = nid(c);
                E(c, OpLoad, tid(S, t), val, var);
            }
            declare(c, d->p[k].name, d->p[k].nlen, SYM_VALUE, mkval(t, val));
        }
        // outputs
        if (d->ret){
            Type *rt = f->ret;
            if (rt->k == T_STRUCT){
                c->out_is_struct = 1;
                for (int j = 0; j < rt->s->n && j < 16; j++){
                    c->outs[c->nouts].t = rt->s->mt[j];
                    c->outs[c->nouts].var = io_var(c, rt->s->mt[j], rt->s->mat[j], 1, d->line);
                    if (rt->s->mat[j].builtin == BI_POSITION) pos_out = c->outs[c->nouts].var;
                    c->nouts++;
                }
            } else {
                c->outs[0].t = rt;
                c->outs[0].var = io_var(c, rt, d->ret_at, 1, d->line);
                if (d->ret_at.builtin == BI_POSITION) pos_out = c->outs[0].var;
                c->nouts = 1;
            }
        }
    } else {
        for (int k = 0; k < d->np; k++) declare(c, d->p[k].name, d->p[k].nlen, SYM_VALUE, mkval(f->pt[k], pid[k]));
    }
    // the body: a vertex shader's position written through a function variable, turned on return
    uint32_t pos_tmp = 0;
    if (pos_out && c->stage == ST_VERTEX){
        pos_tmp = pos_out;
        for (int k = 0; k < c->nouts; k++) if (c->outs[k].var == pos_out){ c->outs[k].var = fvar(c, c->outs[k].t, 0); pos_tmp = c->outs[k].var; }
    }
    block_stmts(c, d->body);
    if (!c->dead){
        if (d->stage || f->ret->k == T_VOID){
            if (d->stage && c->nouts) wgsl_fail(c->W, d->line, "%.*s must return a value", d->nlen, d->name);
            E(c, OpReturn);
        } else E(c, OpUnreachable);
    }
    scope_close(c);
    // the position's y turned, before each return of a vertex shader
    Words body = S->fbody;
    if (pos_tmp && pos_tmp != pos_out){
        Words fixed;
        memset(&fixed, 0, sizeof(fixed));
        S->fbody = fixed;
        for (size_t at = 0; at < body.n;){
            uint32_t w0 = body.w[at], len = w0 >> 16, opc = w0 & 0xffff;
            if (opc == OpReturn){
                Type *v4 = t_vec(S, t_f32(S), 4), *f1 = t_f32(S);
                uint32_t p = nid(c), y = nid(c), ny = nid(c), q = nid(c);
                E(c, OpLoad, tid(S, v4), p, pos_tmp);
                E(c, OpCompositeExtract, tid(S, f1), y, p, 1);
                E(c, OpFNegate, tid(S, f1), ny, y);
                E(c, OpCompositeInsert, tid(S, v4), q, ny, p, 1);
                E(c, OpStore, pos_out, q);
            }
            for (uint32_t k = 0; k < len; k++) put(c->W, &S->fbody, body.w[at + k]);
            at += len ? len : 1;
        }
        free(body.w);
        body = S->fbody;
    }
    for (size_t k = 0; k < S->fvars.n; k++) put(c->W, fw, S->fvars.w[k]);
    for (size_t k = 0; k < body.n; k++) put(c->W, fw, body.w[k]);
    OP(S, fw, OpFunctionEnd);
    free(S->fvars.w); free(body.w);
    memset(&S->fvars, 0, sizeof(S->fvars));
    memset(&S->fbody, 0, sizeof(S->fbody));
    // the entry point itself
    if (d->stage){
        f->io = wgsl_alloc(c->W, (size_t)(c->niface + 1) * sizeof(uint32_t));
        memcpy(f->io, c->iface, (size_t)c->niface * sizeof(uint32_t));
        f->nio = c->niface;
        if (d->stage == ST_COMPUTE){
            uint32_t wg[3] = {1, 1, 1};
            Words saved = S->fbody;
            memset(&S->fbody, 0, sizeof(S->fbody));
            for (int k = 0; k < 3; k++) if (d->wg[k]) wg[k] = (uint32_t)const_int(c, d->wg[k], d->line);
            free(S->fbody.w);
            S->fbody = saved;
            OP(S, &S->modes, OpExecutionMode, f->id, 17, wg[0], wg[1], wg[2]);
        } else if (d->stage == ST_FRAGMENT){
            OP(S, &S->modes, OpExecutionMode, f->id, 7);           // OriginUpperLeft
            if (c->uses_frag_depth) OP(S, &S->modes, OpExecutionMode, f->id, 12);   // DepthReplacing
        }
    }
    c->fn = NULL;
    c->stage = 0;
}

// ---- the module ----
void wgsl_emit(Wgsl *W, uint32_t **words, size_t *nwords){
    Cx cx;
    memset(&cx, 0, sizeof(cx));
    Cx *c = &cx;
    c->W = W;
    c->sp.W = W;
    Sp *S = SP(c);
    // the module's names: structs, globals, functions (any order in WGSL)
    c->structs = wgsl_alloc(W, (size_t)(W->nstructs + 1) * sizeof(StructInfo *));
    for (int i = 0; i < W->nstructs; i++){
        StructInfo *si = wgsl_alloc(W, sizeof(StructInfo));
        si->name = W->structs[i].name; si->nlen = W->structs[i].nlen;
        si->decl = &W->structs[i];
        c->structs[i] = si;
        Sym s; memset(&s, 0, sizeof(s));
        s.name = si->name; s.nlen = si->nlen; s.kind = SYM_STRUCT; s.index = i;
        sym_push(c, s);
    }
    c->gl = wgsl_alloc(W, (size_t)(W->nglobals + 1) * sizeof(GlobalInfo));
    for (int i = 0; i < W->nglobals; i++){
        Sym s; memset(&s, 0, sizeof(s));
        s.name = W->globals[i].name; s.nlen = W->globals[i].nlen; s.kind = SYM_GLOBAL; s.index = i;
        sym_push(c, s);
    }
    c->fns = wgsl_alloc(W, (size_t)(W->nfns + 1) * sizeof(FnInfo));
    for (int i = 0; i < W->nfns; i++){
        c->fns[i].d = &W->fns[i];
        c->fns[i].uses = wgsl_alloc(W, (size_t)(W->nglobals + 1));
        c->fns[i].calls = wgsl_alloc(W, (size_t)(W->nfns + 1));
        Sym s; memset(&s, 0, sizeof(s));
        s.name = W->fns[i].name; s.nlen = W->fns[i].nlen; s.kind = SYM_FN; s.index = i;
        sym_push(c, s);
    }
    S->glsl = new_id(S);
    for (int i = 0; i < W->nglobals; i++) resolve_global(c, i);
    for (int i = 0; i < W->nfns; i++) emit_fn(c, i);
    // the entry points: each with its inputs, outputs and every global its calls reach
    uint8_t *reach = wgsl_alloc(W, (size_t)(W->nfns + 1)), *used = wgsl_alloc(W, (size_t)(W->nglobals + 1));
    int *stack = wgsl_alloc(W, (size_t)(W->nfns + 1) * sizeof(int));
    for (int i = 0; i < W->nfns; i++){
        FnInfo *f = &c->fns[i];
        if (!f->d->stage) continue;
        memset(reach, 0, (size_t)W->nfns);
        memset(used, 0, (size_t)W->nglobals);
        int sp = 0;
        stack[sp++] = i; reach[i] = 1;
        while (sp){
            FnInfo *g = &c->fns[stack[--sp]];
            for (int k = 0; k < W->nglobals; k++) if (g->uses[k]) used[k] = 1;
            for (int k = 0; k < W->nfns; k++) if (g->calls[k] && !reach[k]){ reach[k] = 1; stack[sp++] = k; }
        }
        uint32_t *ifc = wgsl_alloc(W, (size_t)(f->nio + W->nglobals + 1) * sizeof(uint32_t));
        int n = 0;
        for (int k = 0; k < f->nio; k++) ifc[n++] = f->io[k];
        for (int k = 0; k < W->nglobals; k++) if (used[k] && !W->globals[k].is_const) ifc[n++] = c->gl[k].var;
        uint32_t model = f->d->stage == ST_COMPUTE ? 5 : f->d->stage == ST_VERTEX ? 0 : 4;
        uint32_t pre[2] = {model, f->id};
        op_str(S, &S->entries, OpEntryPoint, 2, pre, f->d->name, f->d->nlen, n, ifc);
    }
    // the words, in SPIR-V's order
    Words out;
    memset(&out, 0, sizeof(out));
    put(W, &out, 0x07230203); put(W, &out, 0x00010400); put(W, &out, 0); put(W, &out, S->bound + 1); put(W, &out, 0);
    OP(S, &out, OpCapability, 1);                                      // Shader
    if (S->cap_image_query) OP(S, &out, OpCapability, 50);
    if (S->cap_derivative) OP(S, &out, OpCapability, 51);
    { uint32_t id = S->glsl; op_str(S, &out, OpExtInstImport, 1, &id, "GLSL.std.450", 12, 0, NULL); }
    OP(S, &out, OpMemoryModel, 0, 1);                                  // Logical, GLSL450
    Words *parts[] = {&S->entries, &S->modes, &S->names, &S->decos, &S->globals, &S->fns};
    for (size_t p = 0; p < sizeof(parts) / sizeof(parts[0]); p++)
        for (size_t k = 0; k < parts[p]->n; k++) put(W, &out, parts[p]->w[k]);
    out.w[3] = S->bound + 1;
    Words *all[] = {&S->caps, &S->ext, &S->mem, &S->entries, &S->modes, &S->names, &S->decos, &S->globals, &S->fns, &S->fvars, &S->fbody};
    for (size_t p = 0; p < sizeof(all) / sizeof(all[0]); p++) free(all[p]->w);
    free(S->consts); free(S->cc);
    free(c->syms); free(c->scopes); free(c->iface);
    *words = out.w;
    *nwords = out.n;
}
