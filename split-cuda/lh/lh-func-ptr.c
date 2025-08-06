#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <dlfcn.h>

#include "lower-half-api.h"
#include "logging.h"
#include "cudart_apis.h"

static void* Cuda_Fnc_Ptrs[] = {
  NULL,
  FOREACH_FNC(GENERATE_FNC_PTR)
  NULL,
};

void*
lh_dlsym(enum Cuda_Fncs_t fnc)
{
  DLOG(INFO, "LH: Dlsym called with: %d\n", fnc);
  if (fnc < Cuda_Fnc_NULL || fnc > Cuda_Fnc_Invalid) {
    return NULL;
  }
  void *addr = Cuda_Fnc_Ptrs[fnc];
  return addr;
}
