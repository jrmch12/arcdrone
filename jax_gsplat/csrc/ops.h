#pragma once

#include <cuda_runtime.h>
#include <cstddef>

namespace ops {

struct Descriptor {
    unsigned num_points;
    unsigned batch_size;
    dim3 img_shape;        // (W, H, 1)
    float4 intrins;        // (fx, fy, cx, cy)
    float glob_scale;
    float clip_thresh;
    unsigned block_width;
};

namespace render_fwd {

void xla(
    cudaStream_t stream,
    void **buffers,
    const char *opaque,
    std::size_t opaque_len
);

} // namespace render_fwd

} // namespace ops
