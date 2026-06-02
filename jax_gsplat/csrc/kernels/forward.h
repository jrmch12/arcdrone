#pragma once

#include <cuda_runtime.h>

namespace kernels {

__global__ void count_visible(
    const int B,
    const int N,
    const float3 *__restrict__ means3d,
    const float3 *__restrict__ scales,
    const float glob_scale,
    const float4 *__restrict__ quats,
    const float *__restrict__ viewmats,
    const float4 intrins,
    const dim3 img_size,
    const dim3 tile_bounds,
    const unsigned block_width,
    const float clip_thresh,
    int *__restrict__ cam_counts
);

__global__ void project_and_pack(
    const int B,
    const int N,
    const float3 *__restrict__ means3d,
    const float3 *__restrict__ scales,
    const float glob_scale,
    const float4 *__restrict__ quats,
    const float *__restrict__ viewmats,
    const float4 intrins,
    const dim3 img_size,
    const dim3 tile_bounds,
    const unsigned block_width,
    const float clip_thresh,
    const int *__restrict__ cam_offsets,
    int *__restrict__ cam_write_idx,
    float2 *__restrict__ packed_xys,
    float *__restrict__ packed_depths,
    float3 *__restrict__ packed_conics,
    int *__restrict__ packed_num_tiles,
    int *__restrict__ packed_radii,
    float *__restrict__ packed_colors,
    float *__restrict__ packed_opacities,
    const float3 *__restrict__ colors,
    const float *__restrict__ opacities
);

__global__ void expand_intersections(
    const int total_visible,
    const int B,
    const int num_tiles,
    const float2 *__restrict__ packed_xys,
    const float *__restrict__ packed_depths,
    const int *__restrict__ packed_radii,
    const int *__restrict__ cum_tiles,
    const dim3 tile_bounds,
    const unsigned block_width,
    const int *__restrict__ cam_offsets,
    const int *__restrict__ cam_counts,
    int64_t *__restrict__ isect_ids,
    int *__restrict__ flatten_ids
);

__global__ void get_tile_bin_edges(
    const int num_intersects,
    const int64_t *__restrict__ isect_ids_sorted,
    int2 *__restrict__ tile_bins
);

__global__ void batched_rasterize_fwd(
    const int B,
    const dim3 tile_bounds,
    const dim3 img_size,
    const int *__restrict__ flatten_ids_sorted,
    const int2 *__restrict__ tile_bins,
    const float2 *__restrict__ packed_xys,
    const float3 *__restrict__ packed_conics,
    const float *__restrict__ packed_colors,
    const float *__restrict__ packed_opacities,
    const float3 background,
    float3 *__restrict__ out_img
);

} // namespace kernels
