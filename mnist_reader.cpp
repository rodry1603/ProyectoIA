#include "mnist_reader.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

static int be32(int v){
    unsigned char* b = reinterpret_cast<unsigned char*>(&v);
    return (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3];
}

static int read_be32(std::ifstream& f){
    int v = 0;
    f.read(reinterpret_cast<char*>(&v), 4);
    return be32(v);
}

MNISTDataset mnist_load(const std::string& img_path, const std::string& lbl_path, int limit){
    std::ifstream img_f(img_path, std::ios::binary);
    std::ifstream lbl_f(lbl_path, std::ios::binary);

    if(!img_f) throw std::runtime_error("No se pudo abrir: " + img_path);
    if(!lbl_f) throw std::runtime_error("No se pudo abrir: " + lbl_path);

    read_be32(img_f);
    int n_img  = read_be32(img_f);
    int rows   = read_be32(img_f);
    int cols   = read_be32(img_f);
    int pixels = rows * cols;

    read_be32(lbl_f);
    read_be32(lbl_f);

    if(limit <= 0 || limit > n_img) limit = n_img;

    std::cout << "MNIST: " << n_img << " muestras disponibles ("
              << rows << "×" << cols << "), cargando " << limit << "...\n";

    MNISTDataset ds;
    ds.images.reserve(limit);
    ds.labels.reserve(limit);

    for(int i = 0; i < limit; ++i){
        std::vector<float> img(pixels);
        for(int p = 0; p < pixels; ++p){
            unsigned char px = 0;
            img_f.read(reinterpret_cast<char*>(&px), 1);
            img[p] = px / 255.0f;
        }
        ds.images.push_back(std::move(img));

        unsigned char lbl = 0;
        lbl_f.read(reinterpret_cast<char*>(&lbl), 1);
        ds.labels.push_back(static_cast<int>(lbl));
    }

    std::cout << "Cargadas " << ds.images.size() << " muestras\n";
    return ds;
}
