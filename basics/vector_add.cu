#include <stdio.h>
#include <cuda_runtime.h>
#define N 8
__global__ void trial(int *a, int *b, int *c,int num) {

  int idx=threadIdx.x+blockDim.x*blockIdx.x;
  if(idx<N){
    c[idx]=a[idx]+b[idx];
  }
}

int main() {
   
    int a[N]={1,2,3,4,5,6,7,8};
    int b[N]={9,10,11,12,13,14,15,16};
    int c[N];
    int *da,*db,*dc;
    cudaMalloc(&da,N*sizeof(int));
    cudaMalloc(&db,N*sizeof(int));
    cudaMalloc(&dc,N*sizeof(int));
    cudaMemcpy(da,a,N*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(db,b,N*sizeof(int),cudaMemcpyHostToDevice);
    trial<<<2,4>>>(da,db,dc,N);
    cudaDeviceSynchronize();
    cudaMemcpy(c,dc,N*sizeof(int),cudaMemcpyDeviceToHost);
    for(int i=0; i<N; i++){
    printf("%d ", c[i]);
}
printf("\n");
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    return 0;

   
}