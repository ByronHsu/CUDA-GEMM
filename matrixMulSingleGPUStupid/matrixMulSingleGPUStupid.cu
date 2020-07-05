#include <stdio.h>
#include <assert.h>

#include <cuda_runtime.h>

// error handling
#include <helper_cuda.h>
// SDK package
#include <helper_functions.h>


__global__ void matrixMultiply(float *d_A, float *d_B, float *d_C, int wA, int wB){
    
    unsigned int x_index = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y_index = blockDim.y * blockIdx.y + threadIdx.y;


    float sum = 0;
    for(int i = 0; i < wA; i++){
        sum += (d_A[y_index * wA + i] * d_B[x_index + i * wB]); // because hB should be equal to wA
    }
    
    d_C[y_index * wB + x_index] = sum;
}

void init_matrix(float* matrix, int size, float val){
    for(int i = 0; i < size; i++)
        matrix[i] = val;
}

int main(){
    const int dimA_y = 512, dimA_x = 512, dimB_y = 512, dimB_x = 512;

    // const int dimA_y = 16, dimA_x = 16, dimB_y = 16, dimB_x = 16;
    
    dim3 dimsA(dimA_x, dimA_y);
    dim3 dimsB(dimB_x, dimB_y);
    dim3 dimsC(dimB_x, dimA_y);

    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = (float*)(malloc(mem_size_A));
    init_matrix(h_A, size_A, 1.0);

    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = (float*)(malloc(mem_size_B));
    init_matrix(h_B, size_B, 2.0);

    unsigned int size_C = dimsC.x * dimsC.y;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float *h_C = (float*)(malloc(mem_size_C));
    init_matrix(h_C, size_C, 0.0);


    const int block_size = 16;
    dim3 thread(block_size, block_size);
    dim3 grid(dimB_x / block_size, dimA_y / block_size);

    float *d_A, *d_B, *d_C;

    checkCudaErrors(cudaMalloc((void **)(&d_A), mem_size_A));
    checkCudaErrors(cudaMalloc((void **)(&d_B), mem_size_B));
    checkCudaErrors(cudaMalloc((void **)(&d_C), mem_size_C));

    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start, stop;
    cudaStream_t stream;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    checkCudaErrors(cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_C, h_C, mem_size_C, cudaMemcpyHostToDevice, stream));

    checkCudaErrors(cudaEventRecord(start, stream));
    const int niter = 10;

    for(int i = 0; i < niter; i++)
        matrixMultiply <<< grid, thread, 0, stream >>>(d_A, d_B, d_C, dimA_x, dimB_x);

    
    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, stream));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / niter;

    checkCudaErrors(cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));


    printf("Time: %.3f\n", msecPerMatrixMul);

    // checkCudaErrors(cudaDeviceSynchronize());
    // for(int i = 0; i < 10; i++){
    //     printf("%.3f ", h_C[i]);
    // }

    // printf("\n");

    free(h_A);
    free(h_B);
    free(h_C);
    
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
}