#include <stdio.h>
#include <unistd.h>

__global__ void helloCUDA()
{
    printf("Hello, CUDA!\n");
}

int main()
{
    helloCUDA<<<1, 1>>>();
    cudaDeviceSynchronize();
    sleep(10);
    return 0;
}
