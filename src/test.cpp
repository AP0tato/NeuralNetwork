#include "metal/wrapper.hpp"
#include <iostream>

bool multiply_matrices_test(MetalWrapper &gpu);
bool add_arrays_test(MetalWrapper &gpu);

int main(int argc, char const *argv[])
{
    MetalWrapper gpu;

    if(!gpu.isAvailable())
        return 1;

    // add_arrays_test(gpu);
    // multiply_matrices_test(gpu);

    return 0;
}

bool add_arrays_test(MetalWrapper &gpu)
{
    std::vector<float> a = {1, 2, 3};
    std::vector<float> b = {3, 2, 1};
    std::vector<float> out;

    gpu.add_arrays(a, b, out);

    for(size_t i = 0; i < out.size(); i++)
        std::cout << out[i] << " ";
    std::cout << "\n";

    if(out[0] != out[1] && out[1] != out[2] && out[0] != 4)
        return 1;

    return 0;
}

bool multiply_matrices_test(MetalWrapper &gpu)
{
    std::vector<std::vector<float>> a = {
        {1, 2},
        {3, 4}
    };
    std::vector<std::vector<float>> b = {
        {1, 2, 3},
        {4, 5, 6}
    };
    std::vector<std::vector<float>> out;

    // gpu.multiply_matrices(a, b, out);

    for(size_t i = 0; i < out.size(); i++)
    {
        for(size_t j = 0; j < out[i].size(); j++)
        {
            std::cout << out[i][j] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
