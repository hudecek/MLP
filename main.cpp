#include <iostream>
#include <iomanip>
#include "Network.h"
#include "PrepareData.h"
#include "Activations.h"

void eval_network(std::vector<unsigned>& definition, ImageData& data) {
    std::cout << "Preparing data" << std::endl;
    std::cout << "Initiating network" << std::endl;
    Network network(definition, 0.01, 0, 0, false, false, Activations::SIGMOID);
    std::cout << "Training network" << std::endl;
    network.train_patterns(1, 10000, data);
    auto e = network.output_pattern_error(data);
    std::cout << "Final error: " << e << std::endl;
}

void mnist_test() {
    std::string inputs = "../MNIST_DATA/mnist_train_vectors.csv";
    std::string outputs = "../MNIST_DATA/mnist_train_labels.csv";
    std::string test_inputs = "../MNIST_DATA/mnist_test_vectors.csv";
    std::string test_outputs = "../MNIST_DATA/mnist_test_labels.csv";

    std::cout << "Preparing data" << std::endl;
    PrepareData data(inputs, outputs);
    std::vector<unsigned> definition = {data.input_dim, 128, 16, 10};

    std::cout << "Initiating network" << std::endl;
    Network network(definition, 0.01, 0, 0, false, false, Activations::SIGMOID);

    std::cout << "Training network" << std::endl;
    network.train(10, 100, data, 10000);
    network.test(test_inputs, test_outputs, data);
}



int main() {
    mnist_test();
    return 0;
}