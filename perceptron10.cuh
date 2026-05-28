#pragma once

#define NUM_NEURONS 10
#define INPUT_SIZE 784

struct PerceptronModel {
    float* d_weights;
    float* d_biases;
    float* d_outputs;
    float* d_input;
};

void perceptron_init(PerceptronModel& m);
void perceptron_free(PerceptronModel& m);

void perceptron_forward(const PerceptronModel& m, const float* h_input, float* h_outputs);

void perceptron_train(const PerceptronModel& m, const float* h_input, int label, float learning_rate);

void perceptron_save(const PerceptronModel& m, const char* path);
void perceptron_load(const PerceptronModel& m, const char* path);
