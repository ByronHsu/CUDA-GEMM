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

typedef struct{
    // matrixes' content
    float *h_pA; // partial matrix A
    float *h_B; // entire matrix B
    float *h_pC; // partial matrix C
    float *d_pA;
    float *d_B;
    float *d_pC;
    
    // stream for async execution
    cudaStream_t stream;
} TGPUplan;

void init_matrix(float* matrix, int size, float val){
    for(int i = 0; i < size; i++)
        matrix[i] = i;//val;
}

int main(int argc, char* argv[]){

    if(argc < 5){
        printf("Usage: ./matrixMulMultiGPUTiling <GPU_N> <A_height> <A_width> <B_height> <B_width>\n");
        return 0;
    }
    
    const int dimA_y = atoi(argv[2]), dimA_x = atoi(argv[3]), dimB_y = atoi(argv[4]), dimB_x = atoi(argv[5]);
    const int MAX_GPU_COUNT = 32, block_size = 16;
    
//     const int dimA_y = 8, dimA_x = 8, dimB_y = 8, dimB_x = 8;
//     const int MAX_GPU_COUNT = 32, block_size = 2;
    

    TGPUplan plan[MAX_GPU_COUNT];
    
    int GPU_N, Sys_GPU_N;
    GPU_N = atoi(argv[1]);
    checkCudaErrors(cudaGetDeviceCount(&Sys_GPU_N));
    if(GPU_N > Sys_GPU_N){
        printf("GPU count should be less than %d\n", Sys_GPU_N);
    }
    printf("GPU count: %d\n", GPU_N);
    
    dim3 dimsA(dimA_x, dimA_y);
    dim3 dimsB(dimB_x, dimB_y);
    dim3 dimsC(dimB_x, dimA_y);

    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A;
    checkCudaErrors(cudaMallocHost((void**)(&h_A), mem_size_A));
    init_matrix(h_A, size_A, 1.0);
    
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B;
    checkCudaErrors(cudaMallocHost((void**)(&h_B), mem_size_B));
    init_matrix(h_B, size_B, 2.0);

    unsigned int size_C = dimsC.x * dimsC.y;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float *h_C;
    checkCudaErrors(cudaMallocHost((void**)(&h_C), mem_size_C));
    init_matrix(h_C, size_C, 0.0);
    
    for(int i = 0; i < GPU_N; i++){
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cudaStreamCreate(&plan[i].stream));
        
        plan[i].h_B = h_B;
        plan[i].h_pA = h_A + i * (size_A / GPU_N);
        plan[i].h_pC = h_C + i * (size_C / GPU_N);
        
        checkCudaErrors(cudaMalloc((void **)(&plan[i].d_B), mem_size_B));
        checkCudaErrors(cudaMalloc((void **)(&plan[i].d_pA), mem_size_A / GPU_N));
        checkCudaErrors(cudaMalloc((void **)(&plan[i].d_pC), mem_size_C / GPU_N));
    }
    
    // printf("Allocated memory space on each GPU\n");
    
    // cal_time = HtoD + kernel + DtoH
    int niter = 300;
    dim3 threads(block_size, block_size);
    dim3 grid(dimsC.x / block_size, dimsC.y / (block_size * GPU_N));

    cudaEvent_t start, stop;
    
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    
    checkCudaErrors(cudaEventRecord(start));
    
    for(int j = 0; j < niter; j++){
        
        for(int i = 0; i < GPU_N; i++){
            // switch to gpu[i]
            // printf("Now at GPU[%d]\n", i);
            checkCudaErrors(cudaSetDevice(i));

            // transfer data from host to device
            checkCudaErrors(cudaMemcpyAsync(plan[i].d_B, plan[i].h_B, mem_size_B, cudaMemcpyHostToDevice, plan[i].stream));
            checkCudaErrors(cudaMemcpyAsync(plan[i].d_pA, plan[i].h_pA, mem_size_A / GPU_N, cudaMemcpyHostToDevice, plan[i].stream));
            checkCudaErrors(cudaMemcpyAsync(plan[i].d_pC, plan[i].h_pC, mem_size_C / GPU_N, cudaMemcpyHostToDevice, plan[i].stream));
            // printf("GPU[%d]: Transfer data from host to device\n", i);


            matrixMultiply<block_size>  <<< grid, threads, 0, plan[i].stream >>>(plan[i].d_pA, plan[i].d_B, plan[i].d_pC, dimsA.x, dimsB.x);
            // printf("GPU[%d]: Call kernel\n", i);

            checkCudaErrors(cudaMemcpyAsync(plan[i].h_pC, plan[i].d_pC, mem_size_C / GPU_N, cudaMemcpyDeviceToHost, plan[i].stream));
            // printf("GPU[%d]: Transfer data from device to host\n", i);
        }
        // sync devices

        // wait each GPU complete its task
        for(int i = 0; i < GPU_N; i++){
            checkCudaErrors(cudaSetDevice(i));
            cudaStreamSynchronize(plan[i].stream);
        }
        
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
    
//     for(int i = 0; i < size_C; i++){
//         printf("%.3f ", h_C[i]);
//         if((i + 1) % dimsC.x == 0)
//             printf("\n");
//     }
}