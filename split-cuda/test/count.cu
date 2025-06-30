#include <stdio.h>
#include <cuda_runtime.h>
#include <unistd.h> // For sleep function

__global__ void countKernel(int count) {
    printf("Count: %d\n", count);
}

int main() {
    for (int i = 1; i <= 100; i++) {
        // Launch a CUDA kernel to print the current count
        countKernel<<<1, 1>>>(i);

        // Synchronize the device to ensure all threads have finished before the delay
        cudaDeviceSynchronize();

        // Delay of 0.5 seconds
        usleep(500000); // 500,000 microseconds = 0.5 seconds
    }

    // Reset the device and clean up resources
    cudaDeviceReset();

    return 0;
}

