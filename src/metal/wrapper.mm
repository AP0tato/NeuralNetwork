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
        NSString* metallibPath = @"functions.metallib";
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

void MetalWrapper::multiply_matrices(std::vector<std::vector<float>>& a, std::vector<std::vector<float>>& b, std::vector<std::vector<float>>& out)
{
    if (!hasPipeline("multiply_matrices"))
    {
        out.clear();
        return;
    }

    MetalWrapperImpl* state = static_cast<MetalWrapperImpl*>(impl);
    if (a.empty() || b.empty() || a[0].size() != b.size())
    {
        out.clear();
        return;
    }

    const uint32_t rows = a.size();
    const uint32_t inner_dim = a[0].size();
    const uint32_t out_cols = b[0].size();
    out.resize(rows);
    for(std::size_t i = 0; i < rows; i++)
        out[i].resize(out_cols);

    std::vector<float> compressed_a(a.size() * a[0].size());
    std::vector<float> compressed_b(b.size() * b[0].size());
    std::vector<float> compressed_out(rows * out_cols);

    std::size_t k = 0;
    for(std::size_t i = 0; i < a.size(); i++)
        for(std::size_t j = 0; j < a[i].size(); j++)
            compressed_a[k++] = a[i][j];

    k = 0;
    for(std::size_t i = 0; i < b.size(); i++)
        for(std::size_t j = 0; j < b[i].size(); j++)
            compressed_b[k++] = b[i][j];

    const std::size_t count = compressed_out.size();

    id<MTLBuffer> a_buf = [state->device newBufferWithBytes:compressed_a.data()
                            length:compressed_a.size()*sizeof(float)
                            options:MTLResourceStorageModeShared];

    id<MTLBuffer> b_buf = [state->device newBufferWithBytes:compressed_b.data()
                            length:compressed_b.size()*sizeof(float)
                            options:MTLResourceStorageModeShared];

    id<MTLBuffer> out_buf = [state->device newBufferWithLength:count*sizeof(float)
                            options:MTLResourceStorageModeShared];

    std::size_t idx = state->pipelineIndex["multiply_matrices"];
    id<MTLComputePipelineState> pso = state->pipelines[idx];
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
    if (w > count) w = count;
    MTLSize tg = MTLSizeMake(w, 1, 1);

    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    float* outPtr = static_cast<float*>(out_buf.contents);
    std::copy(outPtr, outPtr + count, compressed_out.begin());

    for(std::size_t i = 0; i < count; i++)
        out[i / out_cols][i % out_cols] = compressed_out[i];
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
