#include<stdio.h>
#include<cuda_runtime.h>


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
    cudaMemcpy(da,a,N*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(db,b,N*sizeof(int),cudaMemcpyHostToDevice);
    trial<<<BlocksPerGrid,ThreadsPerBlock>>>(da,db,dc,N);
    cudaDeviceSynchronize();
    cudaMemcpy(c,dc,N*sizeof(int),cudaMemcpyDeviceToHost);
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