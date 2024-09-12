#include <stdio.h>

#include <opencv2/opencv.hpp>

#define CHANNELS 3

__global__ void colortoGrayscaleConvertion(unsigned char *Pout,
                                           unsigned char *Pin, int width,
                                           int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int grayOffset = row * width + col;
        int rgbOffset = grayOffset * CHANNELS;
        unsigned char b = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char r = Pin[rgbOffset + 2];

        Pout[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}

void launch_colortoGrayscaleConvertion(unsigned char *Pout, unsigned char *Pin,
                                       int width, int height) {
    unsigned char *Pout_d, *Pin_d;
    cudaMalloc(&Pout_d, width * height * sizeof(unsigned char));
    cudaMalloc(&Pin_d, width * height * CHANNELS * sizeof(unsigned char));
    cudaMemcpy(Pin_d, Pin, width * height * CHANNELS * sizeof(unsigned char),
               cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    printf("grid: %d %d\n", grid.x, grid.y);
    printf("block: %d %d\n", block.x, block.y);
    colortoGrayscaleConvertion<<<grid, block>>>(Pout_d, Pin_d, width, height);

    cudaMemcpy(Pout, Pout_d, width * height * sizeof(unsigned char),
               cudaMemcpyDeviceToHost);

    cudaFree(Pout_d);
    cudaFree(Pin_d);
}

int main() {
    // read image using cv2
    cv::Mat img = cv::imread("../data/duck.jpg");
    if (img.empty()) {
        printf("Image not found\n");
        return 1;
    }

    int width = img.cols;
    int height = img.rows;

    printf("Input image shape: %d x %d x 3\n", height, width);
    printf("Input number of pixels: %d x 3 = %d\n", height * width,
           height * width * 3);

    unsigned char *Pout, *Pin;
    Pin = img.data;
    Pout = (unsigned char *)malloc(width * height * sizeof(unsigned char));

    printf("input image size: %d\n", img.total() * img.elemSize());

    launch_colortoGrayscaleConvertion(Pout, Pin, width, height);

    // write image using cv2
    cv::Mat grayImg(height, width, CV_8UC1, Pout);

    printf("output image size: %d\n", grayImg.total() * grayImg.elemSize());
    cv::imwrite("../data/duck_gray.jpg", grayImg);

    // free data created
    free(Pout);

    return 0;
}
