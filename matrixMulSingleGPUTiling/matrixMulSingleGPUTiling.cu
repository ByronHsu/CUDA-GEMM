#include <stdio.h>
#include <assert.h>

#include <cuda_runtime.h>

// error handling
#include <helper_cuda.h>
// SDK package
#include <helper_functions.h>


template <int block_size> __global__ void matrixMultiply(float *d_A, float *d_B, float *d_C, int wA, int wB){
    
    unsigned int x_index = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y_index = blockDim.y * blockIdx.y + threadIdx.y;

    __shared__ float blockA[block_size][block_size];
    __shared__ float blockB[block_size][block_size];

    float Csub = 0;

    int tx = threadIdx.x, ty = threadIdx.y;

    // __syncthreads() acts as a barrier at which all threads in the block must wait before any is allowed to proceed.
    for(int i = 0; i < wA / block_size; i++){

        blockA[ty][tx] = d_A[y_index * wA + (i * block_size + tx)];
        blockB[ty][tx] = d_B[(i * block_size + ty) * wB + x_index];

        __syncthreads();
        
        for(int j = 0; j < block_size; j++){
            Csub += (blockA[ty][j] * blockB[j][tx]);
        }
        __syncthreads();
    }

    d_C[y_index * wA + x_index] = Csub; //blockC[y_index % blockDim.y][x_index % blockDim.x];

}

void init_matrix(float* matrix, int size, float val){
    for(int i = 0; i < size; i++)
        matrix[i] = val;
}

int main(){
    // matrix A, B, C
    // C = A * B
    // GPU METHOD
    // CPU METHOD
    // VERIFY RESULT

    const int dimA_y = 320, dimA_x = 320, dimB_y = 320, dimB_x = 640;

    // const int dimA_y = 4, dimA_x = 4, dimB_y = 4, dimB_x = 4;
    
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

    // printf("123\n");

    const int block_size = 32;
    dim3 threads(block_size, block_size);
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

    checkCudaErrors(cudaEventRecord(start, stream));
    



    const int niter = 300;

    for(int i = 0; i < niter; i++){
        checkCudaErrors(cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream));
        checkCudaErrors(cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream));
        checkCudaErrors(cudaMemcpyAsync(d_C, h_C, mem_size_C, cudaMemcpyHostToDevice, stream));
        
        matrixMultiply<block_size> <<< grid, threads, 0, stream >>>(d_A, d_B, d_C, dimA_x, dimB_x);
        
        checkCudaErrors(cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));
        checkCudaErrors(cudaDeviceSynchronize());
    }

    
    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, stream));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / niter;
    double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) *
                               static_cast<double>(dimsA.y) *
                               static_cast<double>(dimsB.x);
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) /
                       (msecPerMatrixMul / 1000.0f);
    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops," \
        " WorkgroupSize= %u threads/block\n",
        gigaFlops,
        msecPerMatrixMul,
        flopsPerMatrixMul,
        threads.x * threads.y);

    

    
    // for(int i = 0; i < size_C; i++){
    //     printf("%.3f ", h_C[i]);
    // }

    printf("\n");

    free(h_A);
    free(h_B);
    free(h_C);
    
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
}