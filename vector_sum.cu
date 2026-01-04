#include <stdio.h>
#include <cuda_runtime.h>

__global__ void add(int *a, int *b, int *c, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}

int main()
{
    const int N = 1024;
    //CUDA runtime APIs use sizes in BYTES, not element counts
    const size_t bytes = N * sizeof(int);

    //host vectors
    int *h_A = new int [N];
    int *h_B = new int [N];
    int *h_C = new int [N];

    //initialize data
    for (int i = 0; i < N; i++)
    {
        h_A[i]  = i;
        h_B[i]  = i * 2;
    }

    //device vectors
    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    //copy data from host to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    //kernel launch configuration
    int threadsPerBlock = 256;
    // Integer ceiling division. blocksPerGrid = ceil(N / threadsPerBlock)
    // Ensures enough blocks even when N is not divisible by threadsPerBlock.
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("CUDA error: %s\n", cudaGetErrorString(err));

    //copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    //verify top 10 values
    for (int i = 0; i < 10; i++){
        printf("C[%d] = %d\n", i, h_C[i]);
    }

    //free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete [] h_A;
    delete [] h_B;
    delete [] h_C;

    return 0;
}