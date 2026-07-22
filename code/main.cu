#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <iostream>
#include <cstddef>
#include <chrono>
#include <iomanip>
#include "stb_image.h"
#include "stb_image_write.h"
#include <cuda_runtime.h>
#define REQUESTED_CHANNELS 3 // To force RGB output always

using namespace std;

// Simple error-checking macro — wrap every CUDA call in this.
inline void cudaCheck(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line
                   << " -> " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), __FILE__, __LINE__)

unsigned char* read_image(const char* filename, int& width, int& height, int& file_channels) {
    unsigned char* data = stbi_load(filename, &width, &height, &file_channels, REQUESTED_CHANNELS);

    if (data == nullptr) {
        cerr << "Failed to load image: " << filename << endl;
        return nullptr;
    }

    cout << "Image loaded successfully!\n";
    cout << "Width: "  << width  << endl;
    cout << "Height: " << height << endl;
    cout << "Pixels: " << width * height << endl;
    cout << "Channels in file: " << file_channels << endl;
    cout << "Channels loaded: " << REQUESTED_CHANNELS << endl;

    return data;
}

unsigned char* convertToGreyScale(unsigned char* h_image, int width, int height) {
    size_t bytes = static_cast<size_t>(width) * height * sizeof(unsigned char);
    unsigned char* h_grey_image = new unsigned char[bytes];

    for (int i = 0; i < width * height; ++i) {
        int r = h_image[i * REQUESTED_CHANNELS + 0];
        int g = h_image[i * REQUESTED_CHANNELS + 1];
        int b = h_image[i * REQUESTED_CHANNELS + 2];

        // Convert to grayscale using luminosity method
        h_grey_image[i] = static_cast<unsigned char>(0.21f * r + 0.72f * g + 0.07f * b);
    }

    return h_grey_image;
}

void save_image(const char* filename, unsigned char* h_image, int width, int height) {
    if (!stbi_write_jpg(filename, width, height, 1, h_image, 100)) {
        cerr << "Failed to save image: " << filename << endl;
    } else {
        cout << "Image saved successfully: " << filename << endl;
    }
}

__global__ void convertToGreyScaleKernel(unsigned char* d_image, unsigned char* d_grey_image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        int r = d_image[idx * REQUESTED_CHANNELS + 0];
        int g = d_image[idx * REQUESTED_CHANNELS + 1];
        int b = d_image[idx * REQUESTED_CHANNELS + 2];

        // Convert to grayscale using luminosity method
        d_grey_image[idx] = static_cast<unsigned char>(0.21f * r + 0.72f * g + 0.07f * b);
    }
}


int main() {
    const char* filename = "./images/image.jpg";

    int width, height, file_channels;
    unsigned char* h_image = read_image(filename, width, height, file_channels);

    if (h_image == nullptr) {
        return EXIT_FAILURE;
    }

    // CPU-side grayscale conversion
    auto cpu_start = std::chrono::high_resolution_clock::now();
    unsigned char* h_grey_image = convertToGreyScale(h_image, width, height);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_end - cpu_start).count();
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "CPU grayscale conversion time: " << static_cast<double>(cpu_duration_ns) / 1'000'000.0 << " ms" << std::endl;
    save_image("./images/grey_image_cpu.jpg", h_grey_image, width, height);

    // Use REQUESTED_CHANNELS, not file_channels — that's the buffer's real layout.
    size_t bytes = static_cast<size_t>(width) * height * REQUESTED_CHANNELS * sizeof(unsigned char);

    unsigned char* d_image = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_image, bytes));
    CUDA_CHECK(cudaMemcpy(d_image, h_image, bytes, cudaMemcpyHostToDevice));
    cout<<"Image data copied to device memory successfully!" << endl;

    // ... kernel launch goes here ...
    unsigned char* d_grey_image = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_grey_image, bytes/REQUESTED_CHANNELS)); // Allocate memory for grayscale image
    
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    convertToGreyScaleKernel<<<gridDim, blockDim>>>(d_image, d_grey_image, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

    cout << "Kernel launched successfully!" << endl;
    cout << "GPU kernel execution time: " << gpu_ms << " ms" << endl;
    cout << "Kernel execution completed successfully!" << endl;
    cout << "Copying grayscale image data back to host memory..." << endl;

    unsigned char* h_grey_image_gpu = new unsigned char[bytes/REQUESTED_CHANNELS]; // Allocate host memory for grayscale image
    CUDA_CHECK(cudaMemcpy(h_grey_image_gpu, d_grey_image, static_cast<size_t>(width) * height * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    cout<<"Grayscale image data copied back to host memory successfully!" << endl;

    save_image("./images/grey_image_gpu.jpg", h_grey_image_gpu, width, height);
    cout<<"Grayscale image saved successfully!" << endl;


    stbi_image_free(h_image);
    CUDA_CHECK(cudaFree(d_image));

    return 0;
}