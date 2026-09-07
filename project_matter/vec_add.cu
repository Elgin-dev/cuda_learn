#include<stdio.h>
#include<cuda_runtime.h>
#include<chrono>


__global__ void trial(int *a,int *b,int *c,int l){
    int idx=threadIdx.x+blockDim.x*blockIdx.x;
    if(idx<l){
        c[idx]=a[idx]+b[idx];
    }
}

int main(){
    const int N=10000000;
    int ThreadsPerBlock=256;
    int BlocksPerGrid = (N + ThreadsPerBlock - 1) / ThreadsPerBlock;
    int *c=new int[N];
    int *b=new int[N];
    int *a=new int[N];
    
    for(int i=0;i<N;i++){
        a[i]=i;
        b[i]=i*2;
    }

    int *da,*db,*dc;
    cudaMalloc(&da,N*sizeof(int));
    cudaMalloc(&db,N*sizeof(int));
    cudaMalloc(&dc,N*sizeof(int));
   auto start_time_before = std::chrono::high_resolution_clock::now();
    cudaMemcpy(da,a,N*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(db,b,N*sizeof(int),cudaMemcpyHostToDevice);
    auto stop_time_before=std::chrono::high_resolution_clock::now();
  auto duration_before = std::chrono::duration<double, std::milli>(stop_time_before - start_time_before);
      printf("Time before kernel launch: %f ms\n", duration_before.count());
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    trial<<<BlocksPerGrid,ThreadsPerBlock>>>(da,db,dc,N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds;
    cudaEventElapsedTime(&milliseconds,start,stop);
    printf("Kernel time: %f ms\n", milliseconds);
    cudaDeviceSynchronize();
     auto start_time_after = std::chrono::high_resolution_clock::now();
    cudaMemcpy(c,dc,N*sizeof(int),cudaMemcpyDeviceToHost);
     auto stop_time_after = std::chrono::high_resolution_clock::now();
    auto duration_after  = std::chrono::duration<double, std::milli>(stop_time_after - start_time_after);
    printf("Time after kernel launch: %f ms\n", duration_after.count());

    printf("Total time %f ms\n",duration_before.count()+milliseconds+duration_after.count());
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    printf("%d\n",c[0]);
    printf("%d\n",c[N-1]);
    printf("%d\n",c[N/2]);

    delete[] a;
    delete[] b;
    delete[] c;
   

    return 0;
}