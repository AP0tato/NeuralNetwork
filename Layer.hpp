#ifndef LAYER_H

#define LAYER_H
#define E 2.71828182845
#define normalize(x) (1/(1+pow(E, -x)))
#include <iostream>
#include <vector>

class Layer
{
private:
    int nodesIn, nodesOut;
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;

public:
    Layer(int in, int out);
    std::vector<double> calculateOutputs(std::vector<double> in);
};

#endif