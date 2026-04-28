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
    void multiply_matrices(const std::vector<float>& a, 
                       const std::vector<float>& b, 
                       std::vector<float>& out,
                       unsigned int rows, 
                       unsigned int inner_dim, 
                       unsigned int out_cols);
    bool isAvailable() const;
    bool hasPipeline(const std::string& functionName) const;

    // Allocate a buffer on the GPU that lives as long as the Layer
    void* create_persistent_buffer(size_t size);
    
    // Update the contents of an existing GPU buffer
    void update_buffer(void* buffer_ptr, const std::vector<float>& data);
    
    // Multiply using a pre-existing GPU buffer for weights
    void multiply_matrices_persistent(
        void* input_buf, 
        void* weight_buf, 
        void* output_buf,
        unsigned int rows, 
        unsigned int inner_dim, 
        unsigned int out_cols);
    
    void free_buffer(void* buffer_ptr);

    void read_buffer(void* buffer_ptr, std::vector<float>& out_vec);

private:
    void* impl;
};

#endif
