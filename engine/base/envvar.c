// envvar.c - the engine's LR_* environment variables (base/envvar.h).
#include "base/envvar.h"

#include <stdlib.h>
#include <string.h>

const char *lr_env(const char *name){
    const char *v = getenv(name);
    return v && *v ? v : NULL;
}

int lr_env_int(const char *name, int def){
    const char *v = lr_env(name);
    return v ? atoi(v) : def;
}

double lr_env_real(const char *name, double def){
    const char *v = lr_env(name);
    return v ? atof(v) : def;
}

int lr_env_is(const char *name, const char *value){
    const char *v = lr_env(name);
    return v && !strcmp(v, value);
}
