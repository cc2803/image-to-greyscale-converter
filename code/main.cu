#define STB_IMAGE_IMPLEMENTATION
#include <iostream>
#include <cstddef>
#include "stb_image.h"
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

int main() {
    const char* filename = "./images/image.jpg";

    int width, height, file_channels;
    unsigned char* h_image = read_image(filename, width, height, file_channels);

    if (h_image == nullptr) {
        return EXIT_FAILURE;
    }

    // Use REQUESTED_CHANNELS, not file_channels — that's the buffer's real layout.
    size_t bytes = static_cast<size_t>(width) * height * REQUESTED_CHANNELS * sizeof(unsigned char);

    unsigned char* d_image = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_image, bytes));
    CUDA_CHECK(cudaMemcpy(d_image, h_image, bytes, cudaMemcpyHostToDevice));
    cout<<"Image data copied to device memory successfully!" << endl;
    // ... kernel launch goes here ...

    stbi_image_free(h_image);
    CUDA_CHECK(cudaFree(d_image));

    return 0;
}