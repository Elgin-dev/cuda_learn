#include <stdio.h>
#include <cuda_runtime.h>

__global__ void trial1(int *a, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        a[idx] = idx;
    }
}

__global__ void trial2(int *b, int N, int stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        b[idx * stride] = idx;
    }
}

int main()
{
    const int N = 10000000;
    const int stride = 2;
    const int ThreadsPerBlock = 256;

    const int BlocksPerGrid =
        (N + ThreadsPerBlock - 1) / ThreadsPerBlock;

    int *a = new int[N];
    int *b = new int[N * stride];

    int *da;
    int *db;

    cudaMalloc(&da, N * sizeof(int));
    cudaMalloc(&db, N * stride * sizeof(int));

    trial1<<<BlocksPerGrid, ThreadsPerBlock>>>(da, N);
    cudaDeviceSynchronize();


    cudaEvent_t start1, stop1;

    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);

    const int ITERATIONS = 10;

    cudaEventRecord(start1);

    for (int i = 0; i < ITERATIONS; i++)
    {
        trial1<<<BlocksPerGrid, ThreadsPerBlock>>>(da, N);
    }

    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);

    float milliseconds1;

    cudaEventElapsedTime(&milliseconds1, start1,stop1);

    float average1 = milliseconds1 / ITERATIONS;

    printf("Contiguous access: %f ms\n", average1);

    //warmup trial2
    
    trial2<<<BlocksPerGrid, ThreadsPerBlock>>>(da, N,stride);
    cudaDeviceSynchronize();

    cudaEvent_t start2, stop2;

    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    cudaEventRecord(start2);

    for (int i = 0; i < ITERATIONS; i++)
    {
        trial2<<<BlocksPerGrid, ThreadsPerBlock>>>(db, N, stride);
    }

    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);

    float milliseconds2;

    cudaEventElapsedTime(&milliseconds2, start2, stop2);

    float average2 = milliseconds2 / ITERATIONS;

    printf("Strided access (stride=%d): %f ms\n",stride,average2);



    cudaMemcpy(a,da, N * sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy( b, db, N * stride * sizeof(int),cudaMemcpyDeviceToHost);


    printf("\nVerification:\n");
    printf("a[N-1] = %d\n", a[N - 1]);
    printf("b[0] = %d\n", b[0]);
    printf("b[N*stride-2] = %d\n", b[N * stride - 2]);


    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);

    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

    cudaFree(da);
    cudaFree(db);

    delete[] a;
    delete[] b;

    return 0;
}