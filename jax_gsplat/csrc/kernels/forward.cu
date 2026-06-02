#include "forward.h"
#include "helpers.h"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace kernels {

// Grid: (B, ceil(N / 256))
__global__ void count_visible(
    const int B, const int N,
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
) {
    const int cam_id = blockIdx.x;
    const int gauss_id = blockIdx.y * blockDim.x + threadIdx.x;
    if (cam_id >= B || gauss_id >= N) return;

    const float *viewmat = viewmats + cam_id * 16;
    float3 p_world = means3d[gauss_id];
    float3 p_view;
    if (helpers::clip_near_plane(p_world, viewmat, p_view, clip_thresh)) return;

    float3 scale = scales[gauss_id];
    float4 quat = quats[gauss_id];
    float cov3d[6];
    helpers::scale_rot_to_cov3d(scale, glob_scale, quat, cov3d);

    float fx = intrins.x, fy = intrins.y, cx = intrins.z, cy = intrins.w;
    float tan_fovx = 0.5f * img_size.x / fx;
    float tan_fovy = 0.5f * img_size.y / fy;
    float3 cov2d; float comp;
    helpers::project_cov3d_ewa(p_world, cov3d, viewmat, fx, fy, tan_fovx, tan_fovy, cov2d, comp);

    float3 conic; float radius;
    if (!helpers::compute_cov2d_bounds(cov2d, conic, radius)) return;

    float2 center = helpers::project_pix({fx, fy}, p_view, {cx, cy});
    uint2 tile_min, tile_max;
    helpers::get_tile_bbox(center, radius, tile_bounds, tile_min, tile_max, block_width);
    if ((int)(tile_max.x - tile_min.x) * (int)(tile_max.y - tile_min.y) <= 0) return;

    atomicAdd(&cam_counts[cam_id], 1);
}

// Grid: (B, ceil(N / 256))
__global__ void project_and_pack(
    const int B, const int N,
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
) {
    const int cam_id = blockIdx.x;
    const int gauss_id = blockIdx.y * blockDim.x + threadIdx.x;
    if (cam_id >= B || gauss_id >= N) return;

    const float *viewmat = viewmats + cam_id * 16;
    float3 p_world = means3d[gauss_id];
    float3 p_view;
    if (helpers::clip_near_plane(p_world, viewmat, p_view, clip_thresh)) return;

    float3 scale = scales[gauss_id];
    float4 quat = quats[gauss_id];
    float cov3d[6];
    helpers::scale_rot_to_cov3d(scale, glob_scale, quat, cov3d);

    float fx = intrins.x, fy = intrins.y, cx = intrins.z, cy = intrins.w;
    float tan_fovx = 0.5f * img_size.x / fx;
    float tan_fovy = 0.5f * img_size.y / fy;
    float3 cov2d; float comp;
    helpers::project_cov3d_ewa(p_world, cov3d, viewmat, fx, fy, tan_fovx, tan_fovy, cov2d, comp);

    float3 conic; float radius;
    if (!helpers::compute_cov2d_bounds(cov2d, conic, radius)) return;

    float2 center = helpers::project_pix({fx, fy}, p_view, {cx, cy});
    uint2 tile_min, tile_max;
    helpers::get_tile_bbox(center, radius, tile_bounds, tile_min, tile_max, block_width);
    int tile_area = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y);
    if (tile_area <= 0) return;

    int write_pos = cam_offsets[cam_id] + atomicAdd(&cam_write_idx[cam_id], 1);

    packed_xys[write_pos] = center;
    packed_depths[write_pos] = p_view.z;
    packed_conics[write_pos] = conic;
    packed_num_tiles[write_pos] = tile_area;
    packed_radii[write_pos] = (int)radius;

    float3 c = colors[gauss_id];
    packed_colors[write_pos * 3 + 0] = c.x;
    packed_colors[write_pos * 3 + 1] = c.y;
    packed_colors[write_pos * 3 + 2] = c.z;
    packed_opacities[write_pos] = opacities[gauss_id];
}

// Grid: ceil(total_visible / 256)
// Encodes camera ID into sort key: key = ((cam_id * num_tiles + tile_id) << 32) | depth_bits
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
) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_visible) return;

    // Determine which camera this packed element belongs to via binary search
    int cam_id = 0;
    int lo = 0, hi = B - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if ((int)idx >= cam_offsets[mid]) {
            cam_id = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }

    float2 center = packed_xys[idx];
    float depth = packed_depths[idx];
    int radius = packed_radii[idx];

    uint2 tile_min, tile_max;
    helpers::get_tile_bbox(center, (float)radius, tile_bounds, tile_min, tile_max, block_width);

    int start = (idx == 0) ? 0 : cum_tiles[idx - 1];
    int32_t depth_bits = __float_as_int(depth);

    for (unsigned i = tile_min.y; i < tile_max.y; ++i) {
        for (unsigned j = tile_min.x; j < tile_max.x; ++j) {
            int64_t local_tile_id = (int64_t)(i * tile_bounds.x + j);
            int64_t global_tile_id = (int64_t)cam_id * num_tiles + local_tile_id;
            isect_ids[start] = (global_tile_id << 32) | (int64_t)(uint32_t)depth_bits;
            flatten_ids[start] = idx;
            ++start;
        }
    }
}

// Grid: ceil(num_intersects / 256)
__global__ void get_tile_bin_edges(
    const int num_intersects,
    const int64_t *__restrict__ isect_ids_sorted,
    int2 *__restrict__ tile_bins
) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_intersects) return;

    int32_t cur_tile = (int32_t)(isect_ids_sorted[idx] >> 32);
    if (idx == 0 || idx == num_intersects - 1) {
        if (idx == 0)
            tile_bins[cur_tile].x = 0;
        if (idx == num_intersects - 1)
            tile_bins[cur_tile].y = num_intersects;
    }
    if (idx == 0) return;

    int32_t prev_tile = (int32_t)(isect_ids_sorted[idx - 1] >> 32);
    if (prev_tile != cur_tile) {
        tile_bins[prev_tile].y = idx;
        tile_bins[cur_tile].x = idx;
    }
}

// Grid: (B, tiles_x, tiles_y)
// Block: (BLOCK_WIDTH, BLOCK_WIDTH, 1)
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
) {
    auto block = cg::this_thread_block();

    const int cam_id = blockIdx.x;
    if (cam_id >= B) return;

    const int tile_col = blockIdx.y;
    const int tile_row = blockIdx.z;

    int32_t local_tile_id = tile_row * tile_bounds.x + tile_col;
    int32_t global_tile_id = cam_id * (tile_bounds.x * tile_bounds.y) + local_tile_id;

    unsigned px_x = tile_col * blockDim.x + threadIdx.x;
    unsigned px_y = tile_row * blockDim.y + threadIdx.y;
    float px = (float)px_x + 0.5f;
    float py = (float)px_y + 0.5f;

    const int img_pixels = img_size.x * img_size.y;
    int32_t pix_id = px_y * img_size.x + px_x;
    int32_t out_pix_id = cam_id * img_pixels + pix_id;

    bool inside = (px_x < img_size.x && px_y < img_size.y);
    bool done = !inside;

    int2 range = tile_bins[global_tile_id];
    const int block_size = block.size();
    int num_batches = (range.y - range.x + block_size - 1) / block_size;

    constexpr int MAX_BLOCK = 256;
    __shared__ int32_t id_batch[MAX_BLOCK];
    __shared__ float3 xy_opacity_batch[MAX_BLOCK];
    __shared__ float3 conic_batch[MAX_BLOCK];

    float T = 1.f;
    int tr = block.thread_rank();
    float3 pix_out = {0.f, 0.f, 0.f};

    for (int b = 0; b < num_batches; ++b) {
        if (__syncthreads_count(done) >= block_size)
            break;

        int batch_start = range.x + block_size * b;
        int load_idx = batch_start + tr;
        if (load_idx < range.y) {
            int flat_id = flatten_ids_sorted[load_idx];
            id_batch[tr] = flat_id;
            const float2 xy = packed_xys[flat_id];
            const float opac = packed_opacities[flat_id];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = packed_conics[flat_id];
        }

        block.sync();

        int batch_size = min(block_size, range.y - batch_start);
        for (int t = 0; (t < batch_size) && !done; ++t) {
            const float3 conic = conic_batch[t];
            const float3 xy_opac = xy_opacity_batch[t];
            const float opac = xy_opac.z;
            const float2 delta = {xy_opac.x - px, xy_opac.y - py};
            const float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                        conic.z * delta.y * delta.y) +
                                conic.y * delta.x * delta.y;
            const float alpha = min(0.999f, opac * __expf(-sigma));
            if (sigma < 0.f || alpha < 1.f / 255.f)
                continue;

            const float next_T = T * (1.f - alpha);
            if (next_T <= 1e-4f) {
                done = true;
                break;
            }

            int32_t g = id_batch[t];
            const float vis = alpha * T;
            pix_out.x += packed_colors[g * 3 + 0] * vis;
            pix_out.y += packed_colors[g * 3 + 1] * vis;
            pix_out.z += packed_colors[g * 3 + 2] * vis;
            T = next_T;
        }
    }

    if (inside) {
        float3 final_color;
        final_color.x = pix_out.x + T * background.x;
        final_color.y = pix_out.y + T * background.y;
        final_color.z = pix_out.z + T * background.z;
        out_img[out_pix_id] = final_color;
    }
}

} // namespace kernels
