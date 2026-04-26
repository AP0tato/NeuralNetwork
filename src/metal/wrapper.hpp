#ifndef WRAPPER_HPP

#define WRAPPER_HPP

#include <string>
#include <vector>

class MetalWrapper
{
public:
    MetalWrapper();
    explicit MetalWrapper(const std::vector<std::string>& functionNames);
    ~MetalWrapper();

    void add_arrays(std::vector<float>& a, std::vector<float>& b, std::vector<float>& out);
    void multiply_matrices(std::vector<std::vector<float>>& a, std::vector<std::vector<float>>& b, std::vector<std::vector<float>>& out);
    bool isAvailable() const;
    bool hasPipeline(const std::string& functionName) const;

private:
    void* impl;
};

#endif
