// file.h - a whole file read into memory: what every loader of the engine starts with
// (glTF, textures, PNG, JPEG, HDR).
#ifndef LR_FILE_H
#define LR_FILE_H

#include <stddef.h>

// The bytes of the file at path, *n of them (a file of 0 bytes: a buffer all the same); NULL
// (and *n = 0) if it cannot be opened or read, or memory runs out. Free with free().
unsigned char *file_read_all(const char *path, size_t *n);

#endif
