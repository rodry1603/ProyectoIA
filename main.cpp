#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <random>
#include <string>

#include "mnist_reader.h"
#include "perceptron10.cuh"

static char pixel_to_ascii(float v){
    if(v > 0.85f) return '@';
    if(v > 0.60f) return '#';
    if(v > 0.35f) return '*';
    if(v > 0.10f) return '.';
    return ' ';
}

static void print_image(const std::vector<float>& img){
    std::cout << "------------------------------\n";
    for(int r = 0; r < 28; ++r){
        std::cout << "|";
        for(int c = 0; c < 28; ++c)
            std::cout << pixel_to_ascii(img[r * 28 + c]);
        std::cout << "|\n";
    }
    std::cout << "------------------------------\n";
}

static int argmax(const float* v, int n){
    return static_cast<int>(std::max_element(v, v + n) - v);
}

static void print_outputs(const float* out){
    std::cout << "  Salidas de neuronas:\n";
    for(int i = 0; i < NUM_NEURONS; ++i){
        std::cout << "  [" << i << "] " << (out[i] > 0.5f ? "1" : "0") << "\n";
    }
}


static void evaluate(const PerceptronModel& model, const MNISTDataset& ds, bool show_images, int  show_limit){
    int correct = 0;
    int shown = 0;

    float outputs[NUM_NEURONS];

    int confusion[10][10] = {};

    for(int i = 0; i < static_cast<int>(ds.images.size()); ++i){
        perceptron_forward(model, ds.images[i].data(), outputs);

        int pred = argmax(outputs, NUM_NEURONS);
        int truth = ds.labels[i];

        if(pred == truth) ++correct;
        confusion[truth][pred]++;

        if(show_images && shown < show_limit){
            std::cout << "\n Muestra " << i << "\n";
            print_image(ds.images[i]);
            print_outputs(outputs);
            std::cout << "  Etiqueta real : " << truth << "\n";
            std::cout << "  Prediccion    : " << pred
                      << (pred == truth ? "  correcta" : "  incorrecta") << "\n";
            ++shown;
        }
    }

    float acc = 100.0f * correct / static_cast<float>(ds.images.size());
 
    std::cout << " Correctas  : " << std::setw(5) << correct << " / " << ds.images.size() << "  \n";
    std::cout << "  Accuracy   : " << std::fixed << std::setprecision(2) << std::setw(7) << acc << " %      \n";

    std::cout << "\n  Matriz de confusion (fila=real, col=pred):\n";
    std::cout << "     ";
    for(int c = 0; c < 10; ++c) std::cout << std::setw(5) << c;
    std::cout << "\n";
    for(int r = 0; r < 10; ++r){
        std::cout << "  [" << r << "]";
        for(int c = 0; c < 10; ++c)
            std::cout << std::setw(5) << confusion[r][c];
        std::cout << "\n";
    }
}


static void train_model(PerceptronModel& model){
    int epochs;
    float lr;
    int n_samples;

    std::cout << "\n  Numero de muestras de entrenamiento (max 60000): ";
    std::cin >> n_samples;
    if(n_samples <= 0 || n_samples > 60000) n_samples = 60000;

    std::cout << "  Epocas        : ";
    std::cin >> epochs;

    std::cout << "  Learning rate : ";
    std::cin >> lr;

    MNISTDataset train_ds = mnist_load(
        "data/train-images.idx3-ubyte",
        "data/train-labels.idx1-ubyte",
        n_samples
    );

    std::vector<int> idx(train_ds.images.size());
    std::iota(idx.begin(), idx.end(), 0);

    std::mt19937 rng(std::random_device{}());

    std::cout << "\n  Iniciando entrenamiento \n\n";

    for(int e = 0; e < epochs; ++e){
        std::shuffle(idx.begin(), idx.end(), rng);

        int mistakes = 0;
        float out[NUM_NEURONS];

        for(int i : idx){
            // conteo de errores
            perceptron_forward(model, train_ds.images[i].data(), out);
            if(argmax(out, NUM_NEURONS) != train_ds.labels[i]) ++mistakes;

            perceptron_train(model, train_ds.images[i].data(), train_ds.labels[i], lr);
        }

        float epoch_acc = 100.0f * (n_samples - mistakes) / static_cast<float>(n_samples);
        std::cout << "  Epoca " << std::setw(3) << (e + 1) << "/" << epochs << "  |  Errores: " << std::setw(6) <<
            mistakes << "  |  Acc: " << std::fixed << std::setprecision(2) << epoch_acc << " %\n";
    }

    std::cout << "\n  Entrenamiento completado.\n";
    perceptron_save(model, "mnist_model.bin");
}


int main(){
    PerceptronModel model;
    perceptron_init(model);

  
    std::cout << "  Perceptron MNIST con 10 neuronas\n";
    std::cout << "  Aprendizaje: correccion de error \n";
   

    // carga modelo
    std::cout << "  1. Entrenar nuevo modelo\n";
    std::cout << "  2. Cargar modelo existente (mnist_model.bin)\n";
    std::cout << "\n  Opcion: ";
    int opt; std::cin >> opt;

    if(opt == 1) {
        train_model(model);
    } else {
        perceptron_load(model, "mnist_model.bin");
    }

    // evaluacion
    std::cout << "\n  ¿Que dataset evaluar?\n";
    std::cout << "  1. Entrenamiento (train)\n";
    std::cout << "  2. Prueba (test)\n";
    std::cout << "\n  Opcion: ";
    int ds_opt; std::cin >> ds_opt;

    std::cout << "  Cantidad de muestras a evaluar (0 = todas): ";
    int eval_n; std::cin >> eval_n;

    std::cout << "  Mostrar imagenes ASCII? (1=si / 0=no): ";
    int show_asc; std::cin >> show_asc;

    int show_lim = 0;
    if(show_asc){
        std::cout << "  Cuantas imagenes mostrar?: ";
        std::cin >> show_lim;
    }

    MNISTDataset eval_ds;
    if(ds_opt == 1){
        eval_ds = mnist_load(
            "data/train-images.idx3-ubyte",
            "data/train-labels.idx1-ubyte",
            eval_n
        );
    } else {
        eval_ds = mnist_load(
            "data/t10k-images.idx3-ubyte",
            "data/t10k-labels.idx1-ubyte",
            eval_n
        );
    }

    std::cout << "  Resultados\n";
    evaluate(model, eval_ds, show_asc != 0, show_lim);

    perceptron_free(model);
    return 0;
}
