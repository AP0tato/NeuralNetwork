#include <metal_stdlib>
using namespace metal;

kernel void add_arrays(const device float* a [[buffer(0)]],
					   const device float* b [[buffer(1)]],
					   device float* out [[buffer(2)]],
					   uint gid [[thread_position_in_grid]])
{
	out[gid] = a[gid] + b[gid];
}

kernel void multiply_matrices(const device float* a [[buffer(0)]],
                              const device float* b [[buffer(1)]],
                              device float* out [[buffer(2)]],
                              constant uint& rows [[buffer(3)]],
                              constant uint& inner_dim [[buffer(4)]],
                              constant uint& out_cols [[buffer(5)]],
                              uint gid [[thread_position_in_grid]])
{
    if (gid >= rows * out_cols) return;

    uint row = gid / out_cols;
    uint col = gid % out_cols;

    float sum = 0.0f;
    for (uint k = 0; k < inner_dim; ++k)
    {
        sum += a[row * inner_dim + k] * b[k * out_cols + col];
    }
    out[gid] = sum;
}