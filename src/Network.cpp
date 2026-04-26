#include "Network.hpp"

Layer::Layer(unsigned long num_nodes_in, unsigned long num_nodes_out)
{
    if(!gpu.isAvailable())
    {
        std::cout << "Could not initialize GPU device\n";
        return;
    }
    this->num_nodes_in = num_nodes_in;
    this->num_nodes_out = num_nodes_out;

    weights.resize(num_nodes_in);
    for(size_t i = 0; i < weights.size(); i++)
        weights[i].resize(num_nodes_out);
    biases.resize(num_nodes_out);
}

float Layer::sigmoid(float x)
{
    { return 1/(1+pow(M_E, -x)); }
}

float Layer::sigmoid_derivative(float x)
{
    float activation = sigmoid(x);
    return activation * (1 - activation);
}

float Layer::cost_derivative(float actual, float expected)
{
    return 2 * (actual - expected);
}

float Layer::cost(float actual, float expected)
{
    float error = actual - expected;
    return error * error;
}

std::vector<float> Layer::get_outputs(std::vector<float> in)
{
    std::vector<std::vector<float>> out;
    std::vector<std::vector<float>> in_2d = {in};
    gpu.multiply_matrices(in_2d, weights, out);

    for(size_t i = 0; i < out[0].size(); i++)
        out[0][i] = sigmoid(out[0][i]);

    return out[0];
}