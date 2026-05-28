#pragma once

#include <vector>
#include <string>

struct MNISTDataset {
    std::vector<std::vector<float>> images;
    std::vector<int> labels;
};

MNISTDataset mnist_load(const std::string& images_path, const std::string& labels_path, int limit = 0);
