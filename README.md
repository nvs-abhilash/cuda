# Learning CUDA

### 1. Installation of CUDA

I installed it using ubuntu drivers, which worked fine.

```bash
sudo ubuntu-drivers install
sudo apt install nvidia-cuda-toolkit
```

Then reboot your device for changes to take affect.

### 2. Running

Create a bin folder to keep all the output binaries in a single folder.


To run C code

```bash
gcc 00_vector_addition.c -o bin/00.out && ./bin/00.out
```

To run CUDA code

```bash
nvcc 00_vector_addition.cu -o bin/00.out && ./bin/00.out
```
