//Scalar multiplication of a vector. It receives as input
//parameters a float number and a float array, and executes the multiplication of each
//element of the array by the given number.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void mult(const float *input,
                     float scalar,
                     float *output,
                     int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
        output[tid] = scalar * input[tid];
}

int main()
{
    int N;
    float scalar;

    // Ask for array size
    printf("Enter number of elements: ");
    scanf("%d", &N);

    // Allocate host memory dynamically
    float *h_input  = new float[N];
    float *h_output = new float[N];

    //CUDA runtime APIs use sizes in BYTES, not element counts
    const size_t bytes = N * sizeof(float);

    //device vectors
    float *d_input;
    float *d_output;
    cudaMalloc((void**)&d_input, bytes);
    cudaMalloc((void**)&d_output, bytes);

    // Ask for array elements
    printf("Enter %d float values:\n", N);
    for (int i = 0; i < N; i++)
        scanf("%f", &h_input[i]);

    //ask for scalar
    printf("Enter the scalar value: \n");
    scanf("%f",&scalar);

    //copy data from host to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    //kernel launch configuration
    int threadsPerBlock = 256;
    // Integer ceiling division. blocksPerGrid = ceil(N / threadsPerBlock)
    // Ensures enough blocks even when N is not divisible by threadsPerBlock.
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    mult<<<blocksPerGrid, threadsPerBlock>>>(d_input, scalar, d_output, N);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("CUDA error: %s\n", cudaGetErrorString(err));

    //copy result back to host
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    //verify top 10 values
    for (int i = 0; i < N; i++){
        printf("%f\n", h_output[i]);
    }

    //free memory
    cudaFree(d_input);
    cudaFree(d_output);
    delete [] h_input;
    delete [] h_output;
    return 0;
}