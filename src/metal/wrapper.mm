#import "wrapper.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

class MetalWrapperImpl
{
public:
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> lib;
    std::vector<id<MTLComputePipelineState>> pipelines;
    std::unordered_map<std::string, std::size_t> pipelineIndex;
};

static void loadPipelines(MetalWrapperImpl* state, const std::vector<std::string>& functionNames)
{
    if (state == nullptr || state->lib == nil || state->device == nil)
    {
        return;
    }

    NSError* error = nil;
    for (const std::string& name : functionNames)
    {
        NSString* functionName = [NSString stringWithUTF8String:name.c_str()];
        id<MTLFunction> fn = [state->lib newFunctionWithName:functionName];
        if (fn == nil)
        {
            NSLog(@"Missing Metal function in library: %@", functionName);
            continue;
        }

        id<MTLComputePipelineState> ps =
            [state->device newComputePipelineStateWithFunction:fn error:&error];
        if (ps == nil)
        {
            NSLog(@"Failed to create pipeline for %@: %@", functionName, error);
            error = nil;
            continue;
        }

        state->pipelineIndex[name] = state->pipelines.size();
        state->pipelines.push_back(ps);
    }
}

MetalWrapper::MetalWrapper() : MetalWrapper({"add_arrays", "multiply_matrices"})
{
}

MetalWrapper::MetalWrapper(const std::vector<std::string>& functionNames) : impl(nullptr)
{
    MetalWrapperImpl* state = new MetalWrapperImpl();
    state->device = MTLCreateSystemDefaultDevice();
    state->queue = (state->device != nil) ? [state->device newCommandQueue] : nil;
    state->lib = nil;

    if (state->device != nil)
    {
        NSError* error = nil;
        NSString* metallibPath = @"build/functions.metallib";
        // Fall back to current directory
        if (![[NSFileManager defaultManager] fileExistsAtPath:metallibPath]) {
            metallibPath = @"functions.metallib";
        }
        NSURL* metallibURL = [NSURL fileURLWithPath:metallibPath];
        state->lib = [state->device newLibraryWithURL:metallibURL error:&error];

        if (state->lib == nil)
        {
            NSLog(@"Failed to load metallib: %@", error);
        }
    }

    impl = state;
    loadPipelines(state, functionNames);
}

MetalWrapper::~MetalWrapper()
{
    MetalWrapperImpl* state = static_cast<MetalWrapperImpl*>(impl);
    delete state;
    impl = nullptr;
}

void MetalWrapper::add_arrays(std::vector<float>& a, std::vector<float>& b, std::vector<float>& out)
{
    @autoreleasepool {
        if (!hasPipeline("add_arrays"))
        {
            out.clear();
            return;
        }

        MetalWrapperImpl* state = static_cast<MetalWrapperImpl*>(impl);
        const std::size_t count = (a.size() < b.size()) ? a.size() : b.size();
        if(count == 0)
        {
            out.clear();
            return;
        }

        out.resize(count);

        id<MTLBuffer> a_buf = [state->device newBufferWithBytes:a.data()
                                length:count*sizeof(float)
                                options:MTLResourceStorageModeShared];

        id<MTLBuffer> b_buf = [state->device newBufferWithBytes:b.data()
                                length:count*sizeof(float)
                                options:MTLResourceStorageModeShared];

        id<MTLBuffer> out_buf = [state->device newBufferWithLength:count*sizeof(float)
                                options:MTLResourceStorageModeShared];

        std::size_t idx = state->pipelineIndex["add_arrays"];
        id<MTLComputePipelineState> pso = state->pipelines[idx];
        id<MTLCommandBuffer> cmd = [state->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:pso];
        [enc setBuffer:a_buf offset:0 atIndex:0];
        [enc setBuffer:b_buf offset:0 atIndex:1];
        [enc setBuffer:out_buf offset:0 atIndex:2];

        MTLSize grid = MTLSizeMake(count, 1, 1);
        NSUInteger w = pso.maxTotalThreadsPerThreadgroup;
        if (w > count) w = count;
        MTLSize tg = MTLSizeMake(w, 1, 1);

        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        float* outPtr = static_cast<float*>(out_buf.contents);
        std::copy(outPtr, outPtr + count, out.begin());
    }
}

void MetalWrapper::multiply_matrices(const std::vector<float>& a, 
                                     const std::vector<float>& b, 
                                     std::vector<float>& out,
                                     unsigned int rows, 
                                     unsigned int inner_dim, 
                                     unsigned int out_cols) 
{
    @autoreleasepool {
        MetalWrapperImpl* state = static_cast<MetalWrapperImpl*>(impl);
        uint32_t count = rows * out_cols;
        out.assign(count, 0.0f);

        id<MTLBuffer> a_buf = [state->device newBufferWithBytes:a.data() length:a.size() * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_buf = [state->device newBufferWithBytes:b.data() length:b.size() * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buf = [state->device newBufferWithBytes:out.data() length:count * sizeof(float) options:MTLResourceStorageModeShared];

        id<MTLComputePipelineState> pso = state->pipelines[state->pipelineIndex["multiply_matrices"]];
        id<MTLCommandBuffer> cmd = [state->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:pso];
        [enc setBuffer:a_buf offset:0 atIndex:0];
        [enc setBuffer:b_buf offset:0 atIndex:1];
        [enc setBuffer:out_buf offset:0 atIndex:2];
        [enc setBytes:&rows length:sizeof(rows) atIndex:3];
        [enc setBytes:&inner_dim length:sizeof(inner_dim) atIndex:4];
        [enc setBytes:&out_cols length:sizeof(out_cols) atIndex:5];

        MTLSize grid = MTLSizeMake(count, 1, 1);
        NSUInteger w = pso.maxTotalThreadsPerThreadgroup;
        MTLSize tg = MTLSizeMake(std::min((NSUInteger)count, w), 1, 1);

        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(out.data(), out_buf.contents, count * sizeof(float));
    }
}

bool MetalWrapper::isAvailable() const
{
    const MetalWrapperImpl* state = static_cast<const MetalWrapperImpl*>(impl);
    return state != nullptr && state->device != nil;
}

bool MetalWrapper::hasPipeline(const std::string& functionName) const
{
    const MetalWrapperImpl* state = static_cast<const MetalWrapperImpl*>(impl);
    if (state == nullptr)
    {
        return false;
    }

    return state->pipelineIndex.find(functionName) != state->pipelineIndex.end();
}

void* MetalWrapper::create_persistent_buffer(size_t size) 
{
    MetalWrapperImpl* state = static_cast<MetalWrapperImpl*>(impl);
    id<MTLBuffer> buf = [state->device newBufferWithLength:size * sizeof(float) 
                                                  options:MTLResourceStorageModeShared];
    return (__bridge_retained void*)buf;
}

// Update weight contents instead of recreating
void MetalWrapper::update_buffer(void* buffer_ptr, const std::vector<float>& data) 
{
    if (!buffer_ptr) return; // Guard against nulls
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buffer_ptr;
    memcpy(buf.contents, data.data(), data.size() * sizeof(float));
}

void MetalWrapper::free_buffer(void* buffer_ptr) {
    if (buffer_ptr) {
        id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)buffer_ptr;
        buf = nil; // Buffer is released here
    }
}

void MetalWrapper::multiply_matrices_persistent(
    void* input_buf, 
    void* weight_buf, 
    void* output_buf,
    unsigned int rows, 
    unsigned int inner_dim, 
    unsigned int out_cols) 
{
    @autoreleasepool {
        MetalWrapperImpl* state = static_cast<MetalWrapperImpl*>(impl);
        uint32_t count = rows * out_cols;

        // Input and Output still need temporary buffers (they change every sample)
        id<MTLBuffer> in_b = (__bridge id<MTLBuffer>)input_buf;
        id<MTLBuffer> we_b = (__bridge id<MTLBuffer>)weight_buf;
        id<MTLBuffer> ou_b = (__bridge id<MTLBuffer>)output_buf;

        id<MTLComputePipelineState> pso = state->pipelines[state->pipelineIndex["multiply_matrices"]];
        id<MTLCommandBuffer> cmd = [state->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:pso];
        [enc setBuffer:in_b offset:0 atIndex:0];
        [enc setBuffer:we_b offset:0 atIndex:1]; // Use persistent buffer
        [enc setBuffer:ou_b offset:0 atIndex:2];
        [enc setBytes:&rows length:sizeof(rows) atIndex:3];
        [enc setBytes:&inner_dim length:sizeof(inner_dim) atIndex:4];
        [enc setBytes:&out_cols length:sizeof(out_cols) atIndex:5];

        [enc dispatchThreads:MTLSizeMake(count, 1, 1) 
             threadsPerThreadgroup:MTLSizeMake(std::min((NSUInteger)count, pso.maxTotalThreadsPerThreadgroup), 1, 1)];
        
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void MetalWrapper::read_buffer(void* buffer_ptr, std::vector<float>& out_vec)
{
    if (!buffer_ptr) return;
    
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buffer_ptr;
    
    // Ensure the output vector is the correct size
    size_t num_elements = buf.length / sizeof(float);
    out_vec.resize(num_elements);
    
    // Copy from GPU-accessible memory to the CPU vector
    memcpy(out_vec.data(), buf.contents, buf.length);
}
