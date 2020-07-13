#include <stdio.h>
#include <assert.h>

#include <cuda_runtime.h>

// error handling
#include <helper_cuda.h>
// SDK package
#include <helper_functions.h>

#define MAX_GPU_COUNT 32

typedef struct{
    float *d_A;
    float *d_B;
    float *d_C;
    cudaStream_t stream;
    bool *hasRead_A;
    bool *hasRead_B;
} GPUPlan;

template <int block_size> __global__ void matrixMultiply(float *d_A, float *d_B, float *d_C, int wA, int wB, unsigned int offsetX, unsigned int offsetY){
    
    unsigned int x_index = blockDim.x * blockIdx.x + threadIdx.x + offsetX;
    unsigned int y_index = blockDim.y * blockIdx.y + threadIdx.y + offsetY;

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

__global__ void print_gpu_content(float* d_A, float* d_B, float* d_C, int wA, int hA, int wB, int hB, int wC, int hC, unsigned int offsetX, unsigned int offsetY){
    printf("At offsetX: %d, offsetY: %d\n", offsetX, offsetY);
    printf("d_A:\n");
    for(int i = 0; i < hA; i++){
        for(int j = 0; j < wA; j++)
            printf("%5.1f ", d_A[i * wA + j]);
        printf("\n");
    }

    printf("d_B:\n");
    for(int i = 0; i < hB; i++){
        for(int j = 0; j < wB; j++)
            printf("%5.1f ", d_B[i * wB + j]);
        printf("\n");
    }
    
    printf("d_C:\n");
    for(int i = 0; i < hC; i++){
        for(int j = 0; j < wC; j++)
            printf("%5.1f ", d_C[i * wC + j]);
        printf("\n");
    }
}

void print_cpu_content(float* h_A, float* h_B, float* h_C, int wA, int hA, int wB, int hB, int wC, int hC){
    printf("h_A:\n");
    for(int i = 0; i < hA; i++){
        for(int j = 0; j < wA; j++)
            printf("%5.1f ", h_A[i * wA + j]);
        printf("\n");
    }

    printf("h_B:\n");
    for(int i = 0; i < hB; i++){
        for(int j = 0; j < wB; j++)
            printf("%5.1f ", h_B[i * wB + j]);
        printf("\n");
    }
    
    printf("h_C:\n");
    for(int i = 0; i < hC; i++){
        for(int j = 0; j < wC; j++)
            printf("%5.1f ", h_C[i * wC + j]);
        printf("\n");
    }
}

void init_matrix(float* matrix, int size, float val){
    for(int i = 0; i < size; i++)
        matrix[i] = i;//val;
}

int main(int argc, char* argv[]){
    if(argc < 5){
        printf("Usage: ./matrixMulMultiGPUTiling <GPU_N> <A_height> <A_width> <B_height> <B_width>\n");
        return 0;
    }
    int GPU_N, Sys_GPU_N;
    GPU_N = atoi(argv[1]);
    checkCudaErrors(cudaGetDeviceCount(&Sys_GPU_N));
    if(GPU_N > Sys_GPU_N){
        printf("GPU count should be less than %d\n", Sys_GPU_N);
    }
    printf("GPU count: %d\n", GPU_N);

    
    const int dimA_y = atoi(argv[2]), dimA_x = atoi(argv[3]), dimB_y = atoi(argv[4]), dimB_x = atoi(argv[5]);
    const int block_size = 32;
    
    GPUPlan plan[MAX_GPU_COUNT];
    
    dim3 dimsA(dimA_x, dimA_y);
    dim3 dimsB(dimB_x, dimB_y);
    dim3 dimsC(dimB_x, dimA_y);
    
    float *h_A, *h_B, *h_C;
    
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    checkCudaErrors(cudaMallocHost((void**)(&h_A), mem_size_A));
    init_matrix(h_A, size_A, 1.0);
    
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    checkCudaErrors(cudaMallocHost((void**)(&h_B), mem_size_B));
    init_matrix(h_B, size_B, 2.0);

    unsigned int size_C = dimsC.x * dimsC.y;
    unsigned int mem_size_C = sizeof(float) * size_C;
    checkCudaErrors(cudaMallocHost((void**)(&h_C), mem_size_C));
    init_matrix(h_C, size_C, 0.0);
    
    // allocate space for device variable
    for(int i = 0; i < GPU_N; i++){
        checkCudaErrors(cudaSetDevice(i));
        
        checkCudaErrors(cudaStreamCreate(&plan[i].stream));
        
        checkCudaErrors(cudaMalloc((void **)(&plan[i].d_A), mem_size_A));
        checkCudaErrors(cudaMalloc((void **)(&plan[i].d_B), mem_size_B));
        checkCudaErrors(cudaMalloc((void **)(&plan[i].d_C), mem_size_C));
        
        plan[i].hasRead_A = (bool*)malloc(dimsC.y / block_size * sizeof(bool));
        plan[i].hasRead_B = (bool*)malloc(dimsC.x / block_size * sizeof(bool));
    }
    
    int wA = dimsA.x, wB = dimsB.x, wC = dimsC.x;
    int hA = dimsA.y, hB = dimsB.y, hC = dimsC.y;
    
    int gpu_idx = 0;
    
//     print_cpu_content(h_A, h_B, h_C, wA, hA, wB, hB, wC, hC);

    cudaEvent_t start, stop;
    
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    
    checkCudaErrors(cudaEventRecord(start));
    
    int niter = 1;
    dim3 threads(block_size, block_size);
    dim3 grid(1, 1);
    for(int t = 0; t < niter; t++)
    for(int i = 0; i < dimsC.y; i += block_size)
        for(int j = 0; j < dimsC.x; j += block_size){
            checkCudaErrors(cudaSetDevice(gpu_idx));
            
            cudaStream_t curr_stream = plan[gpu_idx].stream;
            
            float *d_A = plan[gpu_idx].d_A, *d_B = plan[gpu_idx].d_B, *d_C = plan[gpu_idx].d_C;
            
            if(!plan[gpu_idx].hasRead_A[i / block_size])
                cudaMemcpy2DAsync(d_A + i * wA, wA * sizeof(float), h_A + i * wA, wA * sizeof(float), wA * sizeof(float), block_size, cudaMemcpyHostToDevice, curr_stream);
            
            if(!plan[gpu_idx].hasRead_B[j / block_size])
                cudaMemcpy2DAsync(d_B + j, hB * sizeof(float), h_B + j, hB * sizeof(float), block_size * sizeof(float), hB, cudaMemcpyHostToDevice, curr_stream);
            
            plan[gpu_idx].hasRead_A[i / block_size] = true;
            plan[gpu_idx].hasRead_B[j / block_size] = true;
            
            unsigned int offsetX = j, offsetY = i;
            matrixMultiply<block_size>  <<< grid, threads, 0, curr_stream >>>(plan[gpu_idx].d_A, plan[gpu_idx].d_B, plan[gpu_idx].d_C, dimsA.x, dimsB.x, offsetX, offsetY);
            
            cudaMemcpy2DAsync(h_C + i * wC + j, wC * sizeof(float), d_C + i * wC + j, wC * sizeof(float), block_size * sizeof(float), block_size, cudaMemcpyDeviceToHost, curr_stream);
            
//             print_gpu_content <<<1, 1, 0, curr_stream>>>(d_A, d_B, d_C, wA, hA, wB, hB, wC, hC, offsetX, offsetY);
            
//             cudaStreamSynchronize(curr_stream);
            
    
            gpu_idx = (gpu_idx + 1) % GPU_N;
        }


    checkCudaErrors(cudaEventRecord(stop));
    
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

    
}