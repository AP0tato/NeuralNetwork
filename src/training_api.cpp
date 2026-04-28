#include "Network.hpp"
#include <cstring>
#include <fstream>

// Global network instance
static Network* g_network = nullptr;

extern "C" {

// Initialize network with architecture
int nn_create_network(unsigned int input_size, const unsigned int* hidden_sizes, unsigned int num_hidden, unsigned int output_size)
{
    try
    {
        if (g_network != nullptr)
            delete g_network;
        
        g_network = new Network();
        
        // Add layers
        unsigned int prev_size = input_size;
        
        // Hidden layers
        for (unsigned int i = 0; i < num_hidden; ++i)
        {
            g_network->add_layer(prev_size, hidden_sizes[i]);
            prev_size = hidden_sizes[i];
        }
        
        // Output layer
        g_network->add_layer(prev_size, output_size);
        
        return 0;
    }
    catch (const std::exception& e)
    {
        return 1;
    }
}

// Train a single batch
int nn_train_batch(const float* batch_inputs, const float* batch_labels, 
                               unsigned int batch_size, unsigned int input_size, unsigned int output_size) {
    try {
        if (g_network == nullptr) return 1;

        for (unsigned int i = 0; i < batch_size; ++i) 
        {
            const float* current_input = batch_inputs + (i * input_size);
            const float* current_label = batch_labels + (i * output_size);
            
            g_network->train_sample_ptr(current_input, current_label, input_size, output_size);
        }
        return 0;
    }
    catch (const std::exception& e) {
        return 1;
    }
}

/**
 * Exports current weights/biases to a binary file.
 */
extern "C" int nn_save_to_bin(const char* filepath) {
    try {
        if (g_network == nullptr) return 1;
        
        std::ofstream file(filepath, std::ios::binary);
        if (!file.is_open()) return 2;

        std::vector<float> params = g_network->get_all_parameters();
        
        // Write the count of floats first, then the raw buffer
        uint32_t count = static_cast<uint32_t>(params.size());
        file.write(reinterpret_cast<const char*>(&count), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(params.data()), count * sizeof(float));
        
        file.close();
        return 0;
    }
    catch (...) {
        return 3;
    }
}

// Set learning rate
int nn_set_learning_rate(float lr)
{
    try
    {
        if (g_network == nullptr)
            return 1;
        
        g_network->set_learning_rate(lr);
        return 0;
    }
    catch (const std::exception& e)
    {
        return 1;
    }
}

// Get learning rate
float nn_get_learning_rate()
{
    if (g_network == nullptr)
        return 0.5f;
    
    return g_network->get_learning_rate();
}

// Cleanup
int nn_destroy_network()
{
    try
    {
        if (g_network != nullptr)
        {
            delete g_network;
            g_network = nullptr;
        }
        return 0;
    }
    catch (const std::exception& e)
    {
        return 1;
    }
}

extern "C" void nn_clear_internal_buffers() 
{
    if (g_network) {
        g_network->clear_buffers();
    }
}

int nn_load_from_bin(const char* filepath) 
{
    try {
        if (g_network == nullptr) return 1;
        
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) return 2;

        uint32_t count;
        file.read(reinterpret_cast<char*>(&count), sizeof(uint32_t));
        
        std::vector<float> params(count);
        file.read(reinterpret_cast<char*>(params.data()), count * sizeof(float));
        file.close();

        g_network->set_all_parameters(params);
        return 0;
    }
    catch (...) {
        return 3;
    }
}

} // extern "C"
