#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <algorithm>
#include <iomanip>
#include <stdexcept>
#include <cmath>
#include <numeric>
#include <string>
#include <omp.h> 

using namespace std;

// Utilidades 

double sigmoide(double x) {
    return 1.0 / (1.0 + exp(-x));
}

uint32_t read_big_endian(ifstream& f) {
    unsigned char bytes[4];
    f.read((char*)bytes, 4);
    return ((uint32_t)bytes[0] << 24) | ((uint32_t)bytes[1] << 16) |
        ((uint32_t)bytes[2] << 8) | (uint32_t)bytes[3];
}

//Capa densa

struct CapaDensa {
    int input_size, output_size;
    vector<double> W, b;

    CapaDensa(int in_size, int out_size)
        : input_size(in_size), output_size(out_size),
        W(out_size* in_size), b(out_size, 0.0)
    {
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<> dist(0, 0.01);
        for (auto& w : W) w = dist(gen);
    }

    vector<double> forward(const vector<double>& input) const {
        vector<double> output(output_size, 0.0);

        // Paralelizamos el cálculo de las neuronas de salida
#pragma omp parallel for
        for (int i = 0; i < output_size; i++) {
            double sum = b[i];
            // Este bucle interno procesa los 784 píxeles
            for (int j = 0; j < input_size; j++) {
                sum += W[i * input_size + j] * input[j];
            }
            output[i] = sigmoide(sum);
        }
        return output;
    }

    void backward(const vector<double>& input,
        const vector<double>& grad_out, double lr)
    {
        // Paralelizamos la actualización de pesos por neurona
#pragma omp parallel for
        for (int i = 0; i < output_size; i++) {
            for (int j = 0; j < input_size; j++) {
                W[i * input_size + j] += lr * grad_out[i] * input[j];
            }
            b[i] += lr * grad_out[i];
        }
    }
};

//Perceptrón 

struct SimpleMLP {
    CapaDensa output;
    double lr;

    SimpleMLP(int input_size, int output_size, double lr_)
        : output(input_size, output_size), lr(lr_) {
    }

    vector<double> forward(const vector<double>& x) const {
        return output.forward(x);
    }

    void backward(const vector<double>& x,
        const vector<double>& target,
        const vector<double>& pred)
    {
        vector<double> grad_out(pred.size());
        for (size_t i = 0; i < pred.size(); i++)
            grad_out[i] = (target[i] - pred[i]) * pred[i] * (1.0 - pred[i]);
        output.backward(x, grad_out, lr);
    }

    double compute_loss(const vector<double>& pred, int label) const {
        double loss = 0.0;
        for (int i = 0; i < (int)pred.size(); i++) {
            double t = (i == label ? 1.0 : 0.0);
            loss += pow(pred[i] - t, 2);
        }
        return loss / pred.size();
    }

    int predict(const vector<double>& x) const {
        auto out = forward(x);
        return (int)(max_element(out.begin(), out.end()) - out.begin());
    }

    double evaluate(const vector<vector<double>>& X, const vector<int>& y) const {
        int correct = 0;
        int n = X.size();

        // ¡Magia aquí! Evaluamos múltiples imágenes al mismo tiempo.
        // reduction(+:correct) asegura que varios núcleos puedan sumar a la variable "correct" sin errores.
#pragma omp parallel for reduction(+:correct)
        for (int i = 0; i < n; i++) {
            if (predict(X[i]) == y[i]) correct++;
        }
        return (double)correct / n * 100.0;
    }

    void fit(const vector<vector<double>>& X, const vector<int>& y,
        int epochs, int batch_size,
        const vector<vector<double>>& X_val = {},
        const vector<int>& y_val = {})
    {
        int n = (int)X.size();
        vector<int> indices(n);
        iota(indices.begin(), indices.end(), 0);

        ofstream log_file("loss_history.txt");
        if (!log_file.is_open()) {
            cerr << "No se pudo abrir loss_history.txt para escribir.\n";
            return;
        }
        log_file << "epoch,train_loss,val_loss\n";

        for (int epoch = 0; epoch < epochs; epoch++) {
            mt19937 g(random_device{}());
            shuffle(indices.begin(), indices.end(), g);
            double total_loss = 0.0;
            int num_batches = 0;

            for (int start = 0; start < n; start += batch_size) {
                int end = min(start + batch_size, n);
                double batch_loss = 0.0;

                // El entrenamiento secuencial se mantiene aquí para evitar corromper los pesos
                for (int i = start; i < end; i++) {
                    int idx = indices[i];
                    auto pred = forward(X[idx]);

                    vector<double> target(10, 0.0);
                    target[y[idx]] = 1.0;

                    batch_loss += compute_loss(pred, y[idx]);
                    backward(X[idx], target, pred);
                }
                total_loss += batch_loss / (end - start);
                num_batches++;
            }

            double avg_train_loss = total_loss / num_batches;
            double val_loss = 0.0;

            if (!X_val.empty()) {
                int val_size = X_val.size();
                // Paralelizamos también el cálculo de pérdida de validación
#pragma omp parallel for reduction(+:val_loss)
                for (int i = 0; i < val_size; i++) {
                    val_loss += compute_loss(forward(X_val[i]), y_val[i]);
                }
                val_loss /= val_size;
            }

            cout << "Epoch " << setw(3) << epoch + 1
                << " - Train Loss: " << fixed << setprecision(4) << avg_train_loss;
            if (!X_val.empty())
                cout << " - Val Loss: " << setprecision(4) << val_loss;
            cout << "\n";

            log_file << epoch + 1 << "," << avg_train_loss << ",";
            if (!X_val.empty()) log_file << val_loss;
            log_file << "\n";
        }
        log_file.close();
    }
};

void load_mnist_images(const string& filename, vector<vector<double>>& images) {
    ifstream f(filename, ios::binary);
    if (!f.is_open()) throw runtime_error("No se pudo abrir " + filename);

    uint32_t magic = read_big_endian(f);
    if (magic != 2051) throw runtime_error("Magic number incorrecto en " + filename);

    uint32_t num = read_big_endian(f);
    uint32_t rows = read_big_endian(f);
    uint32_t cols = read_big_endian(f);
    uint32_t size = rows * cols;

    images.reserve(num);
    vector<unsigned char> buf(size);

    for (uint32_t i = 0; i < num; i++) {
        f.read((char*)buf.data(), size);
        vector<double> img(size);
        for (uint32_t j = 0; j < size; j++)
            img[j] = buf[j] / 255.0;
        images.push_back(move(img));
    }
}

void load_mnist_labels(const string& filename, vector<int>& labels) {
    ifstream f(filename, ios::binary);
    if (!f.is_open()) throw runtime_error("no se abrio " + filename);

    uint32_t magic = read_big_endian(f);
    if (magic != 2049) throw runtime_error("magic number incorrecto " + filename);

    uint32_t num = read_big_endian(f);
    labels.reserve(num);

    for (uint32_t i = 0; i < num; i++) {
        unsigned char label;
        f.read((char*)&label, 1);
        labels.push_back((int)label);
    }
}

int main() {
    vector<vector<double>> train_images, test_images;
    vector<int> train_labels, test_labels;


    try {
        load_mnist_images("C:\\Users\\Lenovo\\Downloads\\sebastian\\train-images-idx3-ubyte", train_images);
        load_mnist_labels("C:\\Users\\Lenovo\\Downloads\\sebastian\\train-labels-idx1-ubyte", train_labels);
        load_mnist_images("C:\\Users\\Lenovo\\Downloads\\sebastian\\t10k-images-idx3-ubyte", test_images);
        load_mnist_labels("C:\\Users\\Lenovo\\Downloads\\sebastian\\t10k-labels-idx1-ubyte", test_labels);
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    omp_set_num_threads(omp_get_max_threads());
    cout << "se usa " << omp_get_max_threads() << " nucleos\n";

    cout << "se cargo:  " << train_images.size() << "imagenes "
        << test_images.size() << " de prueba \n\n";

    const int input_size = 28 * 28;
    const int output_size = 10;
    const int epochs = 5;
    const int minibatch_size = 64;
    const double learning_rate = 0.01;

    SimpleMLP model(input_size, output_size, learning_rate);

    model.fit(train_images, train_labels,
        epochs, minibatch_size,
        test_images, test_labels);

    double train_acc = model.evaluate(train_images, train_labels);

    cout << "\nTrain acc: " << fixed << setprecision(2) << train_acc << "%\n";

    cout << "\n muestra iamgenes\n";
    int aciertos_visuales = 0;

    for (int i = 0; i < 50; i++) {
        int prediccion = model.predict(test_images[i]);
        int respuesta_real = test_labels[i];

        cout << "Imagen " << setw(2) << i
            << " red :" << prediccion << " ";

        if (prediccion == respuesta_real) {
            cout << " | Bien : es un  " << respuesta_real << ")\n";
            aciertos_visuales++;
        }
        else {
            cout << " Error " << respuesta_real << ")\n";
        }
    }
    cout << "de las imagnes se adivinaron " << aciertos_visuales;

    return 0;
}