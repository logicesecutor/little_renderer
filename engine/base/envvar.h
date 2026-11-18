// envvar.h - the engine's LR_* environment variables (docs/USAGE.md has the table of them):
// one way to read them. Each is read where it is used, not kept: bin/bench sets some between
// its cases (-env name=value, LR_GPU_HWRT).
#ifndef LR_ENVVAR_H
#define LR_ENVVAR_H

const char *lr_env(const char *name);                    // its value; NULL when unset or empty
int lr_env_int(const char *name, int def);               // its number (as atoi reads it); def when unset
double lr_env_real(const char *name, double def);        // the same as a real number
int lr_env_is(const char *name, const char *value);      // set to exactly value

#endif
