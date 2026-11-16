// json.h - a small JSON reader (RFC 8259), for glTF files (scene/gltf.c).
//
// The whole text is parsed into a tree at once. Numbers are doubles; strings are UTF-8
// (\uXXXX escapes, surrogate pairs included, turned into UTF-8); an object keeps its keys in
// order (a repeated key: the first one is found).
#ifndef LR_JSON_H
#define LR_JSON_H

#include <stddef.h>

typedef enum { JSON_NULL, JSON_FALSE, JSON_TRUE, JSON_NUMBER, JSON_STRING, JSON_ARRAY, JSON_OBJECT } JsonType;

typedef struct Json {
    JsonType type;
    double num;            // JSON_NUMBER
    char *str;             // JSON_STRING
    int n;                 // JSON_ARRAY / JSON_OBJECT: how many items
    struct Json *items;    // ... the items
    char **keys;           // JSON_OBJECT: item i's key
} Json;

// NULL on a syntax error (a message on stderr, with the line)
Json *json_parse(const char *text, size_t len);
void json_free(Json *j);

// Lookups that accept NULL (and give NULL or the default), so paths can be chained:
// json_num(json_get(json_get(m, "pbrMetallicRoughness"), "metallicFactor"), 1)
const Json *json_get(const Json *obj, const char *key);    // an object's item, or NULL
const Json *json_at(const Json *arr, int i);               // an array's item, or NULL
int json_len(const Json *arr);                             // items of an array (0 if not one)
double json_num(const Json *v, double def);                // a number, else def
int json_int(const Json *v, int def);                       // (def also outside int's range)
const char *json_str(const Json *v, const char *def);      // a string, else def

#endif
