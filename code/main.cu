#define STB_IMAGE_IMPLEMENTATION
#include <iostream>
#include "stb_image.h"

using namespace std;

void read_image(const char* filename, int& width, int& height, int& channels) {
    unsigned char* data = stbi_load(filename, &width, &height, &channels, 3);

    if (data == nullptr) {
        cout << "Failed to load image: " << filename << endl;
        return;
    }

    cout << "Image loaded successfully!\n";
    cout << "Width: " << width << endl;
    cout << "Height: " << height << endl;
    cout << "Channels in file: " << channels << endl;

    stbi_image_free(data);
}

int main() {
    const char* filename = "./images/image.jpg";

    int width, height, channels;
    read_image(filename, width, height, channels);

    return 0;
}