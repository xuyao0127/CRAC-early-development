#ifndef PATCH_TRAMPOLINE_H
#define PATCH_TRAMPOLINE_H

#ifdef __cplusplus
extern "C" {
#endif

void patch_trampoline(void *from_addr, void *to_addr);

#ifdef __cplusplus
}
#endif

#endif
