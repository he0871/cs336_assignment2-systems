```c++
__global__ void kernel(float* x, float* y) {
    // register (usually)
    float a = 0.0f;

    // shared memory = on-chip SRAM
    __shared__ float tile[256];

    int i = threadIdx.x;

    // x/y point to global memory = HBM/VRAM
    tile[i] = x[i];   // HBM -> SRAM

    __syncthreads();

    a = tile[i] * 2;  // SRAM -> register

    y[i] = a;         // register -> HBM
}
```