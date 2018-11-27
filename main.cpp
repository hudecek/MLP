#include <iostream>
#include <iomanip>
#include "Network.h"
#include "PrepareData.h"
#include "Activations.h"


void mnist_test() {
    std::string inputs = "../MNIST_DATA/test_in.csv";
    std::string outputs = "../MNIST_DATA/test_out.csv";
    std::string test_inputs = "../MNIST_DATA/test_in.csv";
    std::string test_outputs = "../MNIST_DATA/test_out.csv";

    std::cout << "Preparing data" << std::endl;
    PrepareData data(inputs, outputs);
    std::vector<unsigned> definition = {2, 2, 2};

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