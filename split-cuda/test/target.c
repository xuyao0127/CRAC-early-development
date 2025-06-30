#define _GNU_SOURCE
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <elf.h>
#include <string.h>
#include <fcntl.h>
/*
extern char **__environ;

void print_environment_variables(void) __attribute__((constructor));

void print_environment_variables(void) {
    char **env = __environ;
    printf("Starting address of __environ: %p\n", (void *)__environ);
    printf("Address of the first environment variable: %p\n", (void *)*env);
    printf("First environment variable: %s\n", *env);
    printf("Environment Variables:\n");
    
    while (*env) {
        printf("Address: %p", (void *)*env);
        printf("--%s\n", *env);
        env++;
    }

    printf("\n ----------------------------\n");
    printf("******* LD_PRELOAD = %s *******\n", getenv("LD_PRELOAD"));
}
*/
static void processArgs(int, const char** );

int main(int argc, char **argv) {
    void *cuda_ptr1 = NULL;
    void *cuda_ptr2 = NULL;
    void *cuda_ptr3 = NULL;
    processArgs(argc, (const char**)argv);

    cudaError_t rc = cudaMalloc(&cuda_ptr1, 436*sizeof(char));
    printf("cudaMalloc returned: %d, cuda_ptr1: %p\n", (int)rc, cuda_ptr1);

    sleep(10);  // give enough time to checkpoint

    rc = cudaMalloc(&cuda_ptr2, 43*sizeof(char));
    printf("cudaMalloc returned: %d, cuda_ptr2: %p\n", (int)rc, cuda_ptr2);

    rc = cudaMalloc(&cuda_ptr3, 1025*sizeof(char));
    printf("cudaMalloc returned: %d, cuda_ptr3: %p\n", (int)rc, cuda_ptr3);

    cudaFree(cuda_ptr1);
    cudaFree(cuda_ptr2);
    cudaFree(cuda_ptr3);
    return 0;
}

static void processArgs(int argc, const char** argv) {
    if (argc > 1) {
        printf("Application was called with the following args: ");
        for (int j = 1; j < argc; j++) {
            printf("%s ", argv[j]);
        }
        printf("\n");
    }
}

