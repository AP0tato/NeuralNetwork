#include "Network.hpp"

Layer::Layer(unsigned long num_nodes_in, unsigned long num_nodes_out) 
{
    this->num_nodes_in = num_nodes_in;
    this->num_nodes_out = num_nodes_out;
    
    // Xavier initialization: weights ~ U(-limit, limit) where limit = sqrt(6 / (in + out))
    float xavier_limit = sqrt(6.0f / (num_nodes_in + num_nodes_out));
    weights.resize(num_nodes_in * num_nodes_out);
    for (size_t i = 0; i < weights.size(); i++) {
        // Simple pseudo-random using sine for deterministic but varied init
        weights[i] = xavier_limit * (2.0f * fmod(sin(i * 12.9898f) * 43758.5453f, 1.0f) - 1.0f);
    }
    
    biases.resize(num_nodes_out, 0.0f);
    
    // Allocate GPU memory ONCE for everything
    weight_gpu_buffer = gpu.create_persistent_buffer(weights.size());
    input_gpu_buffer = gpu.create_persistent_buffer(num_nodes_in);
    output_gpu_buffer = gpu.create_persistent_buffer(num_nodes_out);

    gpu.update_buffer(weight_gpu_buffer, weights);
}

Layer::~Layer() {
    gpu.free_buffer(weight_gpu_buffer);
    gpu.free_buffer(input_gpu_buffer);
    gpu.free_buffer(output_gpu_buffer);
}

float Layer::sigmoid(float x)
{
    // Numerically stable sigmoid
    if (x > 20.0f) return 1.0f;      // Avoid overflow
    if (x < -20.0f) return 0.0f;     // Avoid underflow
    return 1.0f / (1.0f + exp(-x));
}

float Layer::sigmoid_derivative(float x)
{
    return x * (1.0f - x);
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
    // 1. Store inputs for use in update_weights later
    this->inputs = in;

    // 2. Run GPU Math
    gpu.update_buffer(input_gpu_buffer, in);
    gpu.multiply_matrices_persistent(input_gpu_buffer, weight_gpu_buffer, output_gpu_buffer, 
                                     1, num_nodes_in, num_nodes_out);

    // 3. Pull the data back so the CPU knows the "values" (activations)
    std::vector<float> out;
    gpu.read_buffer(output_gpu_buffer, out);
    
    // 4. Add biases and apply activation function
    for(size_t i = 0; i < out.size(); i++) {
        float pre_sig = out[i] + biases[i];
        out[i] = sigmoid(pre_sig);
    }
    this->values = out;

    return out;
}

// ──────────────────────────────────────────────────────────────────────────
// TRAINING METHODS
// ──────────────────────────────────────────────────────────────────────────

void Layer::update_weights(const std::vector<float>& delta, float learning_rate) 
{
    const float GRAD_CLIP = 5.0f;  // Gradient clipping threshold
    
    for (size_t i = 0; i < num_nodes_in; i++) {
        float input_val = inputs[i];
        for (size_t j = 0; j < num_nodes_out; j++) {
            float grad = learning_rate * delta[j] * input_val;
            
            // Clip gradient to prevent explosion
            if (grad > GRAD_CLIP) grad = GRAD_CLIP;
            if (grad < -GRAD_CLIP) grad = -GRAD_CLIP;
            
            weights[i * num_nodes_out + j] -= grad;
        }
    }

    gpu.update_buffer(weight_gpu_buffer, weights);

    // Clip bias updates too
    for (size_t j = 0; j < num_nodes_out; j++) {
        float grad = learning_rate * delta[j];
        if (grad > GRAD_CLIP) grad = GRAD_CLIP;
        if (grad < -GRAD_CLIP) grad = -GRAD_CLIP;
        biases[j] -= grad;
    }
}

std::vector<float> Layer::backward_delta(const std::vector<float>& expected, bool is_output_layer)
{
    // For output layer: delta = 2(a - y) * sigmoid'(z)
    std::vector<float> delta(num_nodes_out);
    
    for(size_t i = 0; i < num_nodes_out; i++)
    {
        float a = values[i];
        delta[i] = cost_derivative(a, expected[i]) * sigmoid_derivative(a);
    }
    
    return delta;
}

std::vector<float> Layer::backward_pass(const std::vector<float>& next_delta, 
                                        const std::vector<float>& next_weights, 
                                        unsigned long next_num_nodes_out) 
{
    std::vector<float> current_delta(num_nodes_out);

    for (size_t i = 0; i < num_nodes_out; i++) {
        float error = 0.0f;
        for (size_t j = 0; j < next_num_nodes_out; j++) {
            // weight from current node 'i' to next node 'j'
            error += next_weights[i * next_num_nodes_out + j] * next_delta[j];
        }
        current_delta[i] = error * sigmoid_derivative(values[i]);
    }
    return current_delta;
}

// ──────────────────────────────────────────────────────────────────────────
// NETWORK IMPLEMENTATION
// ──────────────────────────────────────────────────────────────────────────

Network::Network() = default;

Network::~Network()
{
    for(auto layer : layers)
        delete layer;
}

void Network::add_layer(unsigned long num_nodes_in, unsigned long num_nodes_out)
{
    layers.push_back(new Layer(num_nodes_in, num_nodes_out));
}

std::vector<float> Network::forward_pass(const std::vector<float>& input)
{
    std::vector<float> current = input;
    
    unsigned long i = 0;
    for(auto layer : layers)
    {
        current = layer->get_outputs(current);
    }
    
    return current;
}

void Network::backward_pass(const std::vector<float>& expected_output) 
{
    size_t last = layers.size() - 1;
    
    // Output Layer
    std::vector<float> delta = layers[last]->backward_delta(expected_output, true);
    layers[last]->update_weights(delta, learning_rate);
    
    // Hidden Layers
    for (int i = static_cast<int>(last) - 1; i >= 0; --i) {
        // We pass the 1D weights of the layer ahead (i+1)
        delta = layers[i]->backward_pass(delta, 
                                        layers[i+1]->get_weights(), 
                                        layers[i+1]->get_num_nodes_out());
                                        
        layers[i]->update_weights(delta, learning_rate);
    }
}

void Network::train_sample_ptr(const float* input, const float* expected, size_t in_size, size_t out_size) {
    // Convert only once at the point of entry
    std::vector<float> in_vec(input, input + in_size);
    std::vector<float> exp_vec(expected, expected + out_size);
    
    forward_pass(in_vec);
    backward_pass(exp_vec);
}

void Network::train_batch(const std::vector<std::vector<float>>& batch_inputs, const std::vector<std::vector<float>>& batch_expected_outputs)
{
    // Safety check for matching batch sizes
    if (batch_inputs.size() != batch_expected_outputs.size()) 
    {
        return;
    }

    for (size_t i = 0; i < batch_inputs.size(); ++i)
    {
        // Pass the raw data pointers to the sample trainer
        // This avoids creating new vector copies for every sample in the batch
        train_sample_ptr(
            batch_inputs[i].data(), 
            batch_expected_outputs[i].data(), 
            batch_inputs[i].size(), 
            batch_expected_outputs[i].size()
        );
    }
}

std::string Network::export_model_weights_biases() const {
    std::vector<float> flat_params;
    
    for (Layer* layer : layers) {
        // Grab the flat weights and biases from each layer
        const std::vector<float>& w = layer->get_weights();
        const std::vector<float>& b = layer->get_biases();
        
        flat_params.insert(flat_params.end(), w.begin(), w.end());
        flat_params.insert(flat_params.end(), b.begin(), b.end());
    }

    // Convert the float vector to a raw byte string
    std::string binary_data(
        reinterpret_cast<const char*>(flat_params.data()), 
        flat_params.size() * sizeof(float)
    );
    
    return binary_data;
}

void Network::clear_buffers() 
{
    for (Layer* l : layers) {
        // shrink_to_fit() tells the OS to actually take the memory back
        l->get_values_non_const().clear();
        l->get_values_non_const().shrink_to_fit();

        l->get_inputs_non_const().clear();
        l->get_inputs_non_const().shrink_to_fit();
    }
}

void Network::set_all_parameters(const std::vector<float>& params) 
{
    size_t offset = 0;
    std::cerr << "DEBUG: set_all_parameters called with " << params.size() << " params" << std::endl;
    
    for (size_t layer_idx = 0; layer_idx < layers.size(); layer_idx++) {
        Layer* layer = layers[layer_idx];
        // Calculate sizes based on this specific layer
        size_t w_count = layer->get_num_nodes_in() * layer->get_num_nodes_out();
        size_t b_count = layer->get_num_nodes_out();

        std::cerr << "  Layer " << layer_idx << ": in=" << layer->get_num_nodes_in() 
                  << " out=" << layer->get_num_nodes_out() 
                  << " w_count=" << w_count << " b_count=" << b_count << std::endl;

        // Slice the flat vector
        std::vector<float> w(params.begin() + offset, params.begin() + offset + w_count);
        offset += w_count;
        std::vector<float> b(params.begin() + offset, params.begin() + offset + b_count);
        offset += b_count;

        // Fix NaN values in weights - replace with 0.1 (small random-like value)
        int w_nan_count = 0;
        for (auto& val : w) {
            if (std::isnan(val) || std::isinf(val)) {
                val = 0.1f;
                w_nan_count++;
            }
        }
        if (w_nan_count > 0) {
            std::cerr << "    WARNING: Fixed " << w_nan_count << " NaN/Inf weights" << std::endl;
        }

        // Fix NaN values in biases - replace with 0.0
        int b_nan_count = 0;
        for (auto& val : b) {
            if (std::isnan(val) || std::isinf(val)) {
                val = 0.0f;
                b_nan_count++;
            }
        }
        if (b_nan_count > 0) {
            std::cerr << "    WARNING: Fixed " << b_nan_count << " NaN/Inf biases" << std::endl;
        }

        // Debug: Check fixed values
        if (w.size() >= 3) {
            std::cerr << "    Weights (first 3): " << w[0] << " " << w[1] << " " << w[2] << std::endl;
        }
        if (b.size() >= 3) {
            std::cerr << "    Biases (first 3): " << b[0] << " " << b[1] << " " << b[2] << std::endl;
        }

        // Update CPU and GPU
        layer->set_weights_and_biases(w, b);
        
        // Push to Metal Persistent Buffers
        layer->gpu.update_buffer(layer->get_weight_buffer(), w);
    }
}

void Layer::set_weights_and_biases(const std::vector<float>& w, const std::vector<float>& b) {
    this->weights = w;
    this->biases = b;

    // CRITICAL: Update the persistent GPU buffer so the next 
    // forward pass uses the newly loaded weights
    gpu.update_buffer(weight_gpu_buffer, weights);
    
    // Note: If you have a bias_gpu_buffer, update it here as well.
    // If biases are handled on CPU, the update is already done above.
}
