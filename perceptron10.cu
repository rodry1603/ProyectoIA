#include "perceptron10.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include <vector>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;


static void cuda_check(cudaError_t err, const char* msg){
    if(err != cudaSuccess){
        cerr << "[CUDA ERROR] " << msg << ": " << cudaGetErrorString(err) << "\n";
        exit(EXIT_FAILURE);
    }
}


__global__
void kernel_net_input(const float* __restrict__ input, const float* __restrict__ weights, const float* __restrict__ biases, float* net){
    const int neuron = blockIdx.x;
    const int pixel = threadIdx.x;

    atomicAdd(&net[neuron], input[pixel] * weights[neuron * INPUT_SIZE + pixel]);

    if(pixel == 0){
        net[neuron] += biases[neuron];
    }
}

__global__
void kernel_activation(float* net_and_out){
    const int n = threadIdx.x;
    net_and_out[n] = (net_and_out[n] > 0.0f) ? 1.0f : 0.0f;
}


__global__
void kernel_weight_update(const float* __restrict__ input, float* weights, float* biases, const float* __restrict__ outputs, int label, float lr){
    const int neuron = blockIdx.x;
    const int pixel = threadIdx.x;

    const float target = (neuron == label) ? 1.0f : 0.0f;
    const float error = target - outputs[neuron];

    if(error == 0.0f) return;

    weights[neuron * INPUT_SIZE + pixel] += lr * error * input[pixel];

    if(pixel == 0){
        biases[neuron] += lr * error;
    }
}


void perceptron_init(PerceptronModel& m){
    const int W_SIZE = NUM_NEURONS * INPUT_SIZE * sizeof(float);
    const int B_SIZE = NUM_NEURONS * sizeof(float);
    const int O_SIZE = NUM_NEURONS * sizeof(float);
    const int I_SIZE = INPUT_SIZE * sizeof(float);

    cuda_check(cudaMalloc(&m.d_weights, W_SIZE), "alloc weights");
    cuda_check(cudaMalloc(&m.d_biases, B_SIZE), "alloc biases");
    cuda_check(cudaMalloc(&m.d_outputs, O_SIZE), "alloc outputs");
    cuda_check(cudaMalloc(&m.d_input, I_SIZE), "alloc input");

    srand((unsigned)time(nullptr));
    vector<float> h_w(NUM_NEURONS * INPUT_SIZE);
    vector<float> h_b(NUM_NEURONS, 0.0f);

    for(float& v : h_w){
        v = (static_cast<float>(rand()) / RAND_MAX * 0.1f) - 0.05f;
    }

    cuda_check(cudaMemcpy(m.d_weights, h_w.data(), W_SIZE, cudaMemcpyHostToDevice), "copy weights");
    cuda_check(cudaMemcpy(m.d_biases, h_b.data(), B_SIZE, cudaMemcpyHostToDevice), "copy biases");
}

void perceptron_free(PerceptronModel& m){
    cudaFree(m.d_weights);
    cudaFree(m.d_biases);
    cudaFree(m.d_outputs);
    cudaFree(m.d_input);
}

void perceptron_forward(const PerceptronModel& m, const float* h_input, float* h_outputs)
{
    cuda_check(cudaMemcpy(m.d_input, h_input, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice), "fwd copy input");

    cuda_check(cudaMemset(m.d_outputs, 0, NUM_NEURONS * sizeof(float)), "fwd memset");

    kernel_net_input <<<NUM_NEURONS, INPUT_SIZE>>> (m.d_input, m.d_weights, m.d_biases, m.d_outputs);
    kernel_activation<<<1, NUM_NEURONS>>> (m.d_outputs);

    cuda_check(cudaDeviceSynchronize(), "fwd sync");

    cuda_check(cudaMemcpy(h_outputs, m.d_outputs, NUM_NEURONS * sizeof(float), cudaMemcpyDeviceToHost), "fwd copy outputs");
}

void perceptron_train(const PerceptronModel& m, const float* h_input, int label, float learning_rate)
{
    cuda_check(cudaMemcpy(m.d_input, h_input, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice), "train copy input");

    cuda_check(cudaMemset(m.d_outputs, 0, NUM_NEURONS * sizeof(float)), "train memset");

    kernel_net_input <<<NUM_NEURONS, INPUT_SIZE>>> (m.d_input, m.d_weights, m.d_biases, m.d_outputs);
    kernel_activation<<<1, NUM_NEURONS>>> (m.d_outputs);

    kernel_weight_update<<<NUM_NEURONS, INPUT_SIZE>>> (m.d_input, m.d_weights, m.d_biases, m.d_outputs, label, learning_rate);

    cuda_check(cudaDeviceSynchronize(), "train sync");
}

void perceptron_save(const PerceptronModel& m, const char* path){
    vector<float> h_w(NUM_NEURONS * INPUT_SIZE);
    vector<float> h_b(NUM_NEURONS);

    cudaMemcpy(h_w.data(), m.d_weights, NUM_NEURONS * INPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b.data(), m.d_biases,  NUM_NEURONS * sizeof(float), cudaMemcpyDeviceToHost);

    ofstream f(path, ios::binary);
    if(!f){ cerr << "No se pudo abrir para escritura: " << path << "\n"; return; }

    f.write(reinterpret_cast<char*>(h_w.data()), NUM_NEURONS * INPUT_SIZE * sizeof(float));
    f.write(reinterpret_cast<char*>(h_b.data()), NUM_NEURONS * sizeof(float));
    f.close();

    cout << "Modelo guardado en: " << path << "\n";
}

void perceptron_load(const PerceptronModel& m, const char* path){
    vector<float> h_w(NUM_NEURONS * INPUT_SIZE);
    vector<float> h_b(NUM_NEURONS);

    ifstream f(path, ios::binary);
    if(!f){ cerr << "No se pudo abrir para lectura: " << path << "\n"; exit(1); }

    f.read(reinterpret_cast<char*>(h_w.data()), NUM_NEURONS * INPUT_SIZE * sizeof(float));
    f.read(reinterpret_cast<char*>(h_b.data()), NUM_NEURONS * sizeof(float));
    f.close();

    cudaMemcpy(m.d_weights, h_w.data(), NUM_NEURONS * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(m.d_biases,  h_b.data(), NUM_NEURONS * sizeof(float), cudaMemcpyHostToDevice);

    cout << "Modelo cargado desde: " << path << "\n";
}
