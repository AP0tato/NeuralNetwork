#ifndef NODE_HPP

#define NODE_HPP
#include <math.h>
#include <vector>
#include <iostream>
#include "metal/wrapper.hpp"

#define MAX_PARAMETERS 500

class Layer
{
    private:
    MetalWrapper gpu;
    unsigned long num_nodes_in;
    unsigned long num_nodes_out;
    std::vector<float> biases;
    std::vector<std::vector<float>> weights;

    public:
    Layer(unsigned long num_nodes_in, unsigned long num_nodes_out);
    std::vector<float> get_outputs(std::vector<float> in);
    float cost(float actual, float expected);
    float cost_derivative(float actual, float expected);
    float sigmoid(float x);
    float sigmoid_derivative(float x);
};

class Network
{
    private:
    std::vector<Layer*> layers;
};

#endif
