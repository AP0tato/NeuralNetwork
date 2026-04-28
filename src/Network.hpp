#ifndef NODE_HPP

#define NODE_HPP
#include <math.h>
#include <vector>
#include <iostream>
#include <string>
#include <cctype>
#include <stdexcept>
#include "metal/wrapper.hpp"

class Layer
{
    private:
    void* input_gpu_buffer = nullptr;
    void* output_gpu_buffer = nullptr;
    void* weight_gpu_buffer = nullptr;

    unsigned long num_nodes_in;
    unsigned long num_nodes_out;
    std::vector<float> biases;
    std::vector<float> weights;
    std::vector<float> values;
    std::vector<float> inputs;

    public:
    MetalWrapper gpu;
    Layer(unsigned long num_nodes_in, unsigned long num_nodes_out);

    float cost(float actual, float expected);
    float cost_derivative(float actual, float expected);
    float sigmoid(float x);
    float sigmoid_derivative(float x);

    std::vector<float> get_outputs(std::vector<float> in);
    
    // Training methods
    void update_weights(const std::vector<float>& delta, float learning_rate);
    std::vector<float> backward_delta(const std::vector<float>& expected, bool is_output_layer);
    std::vector<float> backward_pass(const std::vector<float>& next_delta, 
                                     const std::vector<float>& next_weights, 
                                     unsigned long next_num_nodes_out);
    void accumulate_gradients(const std::vector<float>& delta, 
                              std::vector<float>& weight_grads, 
                              std::vector<float>& bias_grads) const;
    
    // Accessors
    void set_weights_and_biases(const std::vector<float>& w, const std::vector<float>& b);

    const std::vector<float>& get_weights() const { return weights; }
    const std::vector<float>& get_biases() const { return biases; }
    const std::vector<float>& get_values() const { return values; }
    const std::vector<float>& get_inputs() const { return inputs; }
    const float get_num_nodes_out() const { return num_nodes_out; }
    const float get_num_nodes_in() const { return num_nodes_in; }
    std::vector<float>& get_values_non_const() { return values; }
    std::vector<float>& get_inputs_non_const() { return inputs; }
    void* get_weight_buffer() { return weight_gpu_buffer; }

    ~Layer();
};

class Network
{
    private:
    std::vector<Layer*> layers;
    float learning_rate = 0.5;

    public:
    Network();
    ~Network();
    
    void add_layer(unsigned long num_nodes_in, unsigned long num_nodes_out);
    std::vector<float> forward_pass(const std::vector<float>& input);
    void backward_pass(const std::vector<float>& expected_output);
    void train_sample_ptr(const float* input, const float* expected, size_t in_size, size_t out_size);
    void train_batch(const std::vector<std::vector<float>>& batch_inputs, const std::vector<std::vector<float>>& batch_expected_outputs);
    
    float get_learning_rate() const { return learning_rate; }
    void set_learning_rate(float lr) { learning_rate = lr; }
    
    const std::vector<Layer*>& get_layers() const { return layers; }
    std::vector<float> get_all_parameters() const {
        std::vector<float> flat_data;
        for (auto layer : layers) {
            const std::vector<float>& w = layer->get_weights();
            const std::vector<float>& b = layer->get_biases();
            flat_data.insert(flat_data.end(), w.begin(), w.end());
            flat_data.insert(flat_data.end(), b.begin(), b.end());
        }
        return flat_data;
    }
    std::string export_model_weights_biases() const;

    void clear_buffers();

    void set_all_parameters(const std::vector<float>& params);
};

#endif
