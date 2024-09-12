#include <stdio.h>

#include <opencv2/opencv.hpp>
#define BLUR_SIZE 7

__global__ void blurKernel(unsigned char *in, unsigned char *out, int w,
                           int h) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < w && row < h) {
        int pixVal = 0;
        int pixels = 0;

        for (int blurRow=-BLUR_SIZE; blurRow < BLUR_SIZE+1; ++blurRow) {
            for (int blurCol=-BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol) {
                int currRow = row + blurRow;
                int currCol = col + blurCol;

                if (currRow >= 0 && currRow < h && currCol >= 0 && currCol < w) {
                    pixVal += in[currRow * w + currCol];
                    pixels++;
                }
            }
        }
        out[row*w + col] = (unsigned char)(pixVal / pixels);
    }
}

void launch_blurKernel(unsigned char *in_h, unsigned char *out_h, int w,
                       int h) {
    unsigned char *in_d, *out_d;
    int size = w * h * sizeof(unsigned char);
    cudaMalloc((void **)&in_d, size);
    cudaMalloc((void **)&out_d, size);
    cudaMemcpy(in_d, in_h, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    blurKernel<<<grid, block>>>(in_d, out_d, w, h);

    cudaMemcpy(out_h, out_d, size, cudaMemcpyDeviceToHost);
    cudaFree(in_d);
    cudaFree(out_d);
}

int main() {
    cv::Mat img = cv::imread("../data/duck_gray.jpg", cv::IMREAD_GRAYSCALE);
    printf("completed loading image\n");
    if (img.empty()) {
        printf("Image not found\n");
        return 1;
    }
    int w = img.cols;
    int h = img.rows;
    unsigned char *in_h = img.data;
    unsigned char *out_h =
        (unsigned char *)malloc(w * h * sizeof(unsigned char));

    launch_blurKernel(in_h, out_h, w, h);

    cv::Mat blurred_img(h, w, CV_8UC1, out_h);
    cv::imwrite("../data/duck_gray_blurred.jpg", blurred_img);

    free(out_h);
    return 0;
}