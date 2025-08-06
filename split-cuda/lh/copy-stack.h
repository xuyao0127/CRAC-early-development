#ifndef COPY_STACK_H
#define COPY_STACK_H

#ifdef __cplusplus
extern "C" {
#endif

char *deepCopyStack(int argc, char **argv, char *argc_ptr, char *argv_ptr,
                   unsigned long dest_argc, char **dest_argv, char *dest_stack,
                   Elf64_auxv_t **auxv_ptr, void **stack_bottom);

#ifdef __cplusplus
}
#endif

#endif
