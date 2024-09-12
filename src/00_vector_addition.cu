#include <stdio.h>
#include <stdlib.h>

__global__ void vecAddKernel(float *A, float *B, float *C, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    C[i] = A[i] + B[i];
  }
}

void vecAdd(float *A_h, float *B_h, float *C_h, int n) {
  int size = n * sizeof(float);
  float *A_d, *B_d, *C_d;

  // Allocate memory in device global memory
  cudaMalloc((void **)&A_d, size);
  cudaMalloc((void **)&B_d, size);
  cudaMalloc((void **)&C_d, size);

  // Copy A_h, B_h to A_d, B_d
  cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

  // Call device kernel to compute sum
  int n_threads = 256;
  int n_blocks = ceil(n / float(n_threads));
  vecAddKernel<<<n_blocks, n_threads>>>(A_d, B_d, C_d, n);

  // Copy C_d to C_h
  cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

  // Free memory from device global memory
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

int main() {
  float *A_h, *B_h, *C_h;
  int n = 100;
  int size = n * sizeof(float);

  // Allocate memory in host memory
  A_h = (float *)malloc(size);
  B_h = (float *)malloc(size);
  C_h = (float *)malloc(size);

  // Initialize A_h, B_h
  for (int i = 0; i < n; i++) {
    A_h[i] = 1.0 * i;
    B_h[i] = 2.0 * i;
  }

  // Compute sum
  vecAdd(A_h, B_h, C_h, n);

  // Print data of C_h
  for (int i = 0; i < n; i++) {
    printf("%.2f ", C_h[i]);
  }

  printf("\n");

  // Free host memory
  free(A_h);
  free(B_h);
  free(C_h);

  return 0;
}