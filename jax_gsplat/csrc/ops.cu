#include "common.h"
#include "ffi.h"
#include "kernels/forward.h"
#include "ops.h"

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cstddef>

static void cub_exclusive_sum(cudaStream_t stream, const int *in, int *out, int n) {
    void *ws = nullptr;
    size_t ws_bytes = 0;
    CUDA_THROW_IF_ERR(cub::DeviceScan::ExclusiveSum(ws, ws_bytes, in, out, n, stream));
    CUDA_THROW_IF_ERR(cudaMallocAsync(&ws, ws_bytes, stream));
    CUDA_THROW_IF_ERR(cub::DeviceScan::ExclusiveSum(ws, ws_bytes, in, out, n, stream));
    CUDA_THROW_IF_ERR(cudaFreeAsync(ws, stream));
}

static void cub_inclusive_sum(cudaStream_t stream, const int *in, int *out, int n) {
    void *ws = nullptr;
    size_t ws_bytes = 0;
    CUDA_THROW_IF_ERR(cub::DeviceScan::InclusiveSum(ws, ws_bytes, in, out, n, stream));
    CUDA_THROW_IF_ERR(cudaMallocAsync(&ws, ws_bytes, stream));
    CUDA_THROW_IF_ERR(cub::DeviceScan::InclusiveSum(ws, ws_bytes, in, out, n, stream));
    CUDA_THROW_IF_ERR(cudaFreeAsync(ws, stream));
}

static void cub_radix_sort_pairs(cudaStream_t stream,
                                  const int64_t *keys_in, int64_t *keys_out,
                                  const int *vals_in, int *vals_out,
                                  int n, int end_bit) {
    void *ws = nullptr;
    size_t ws_bytes = 0;
    CUDA_THROW_IF_ERR(cub::DeviceRadixSort::SortPairs(
        ws, ws_bytes, keys_in, keys_out, vals_in, vals_out, n, 0, end_bit, stream));
    CUDA_THROW_IF_ERR(cudaMallocAsync(&ws, ws_bytes, stream));
    CUDA_THROW_IF_ERR(cub::DeviceRadixSort::SortPairs(
        ws, ws_bytes, keys_in, keys_out, vals_in, vals_out, n, 0, end_bit, stream));
    CUDA_THROW_IF_ERR(cudaFreeAsync(ws, stream));
}

static int read_device_int(cudaStream_t stream, const int *d_ptr) {
    int val = 0;
    cudaStreamSynchronize(stream);
    CUDA_THROW_IF_ERR(cudaMemcpy(&val, d_ptr, sizeof(int), cudaMemcpyDeviceToHost));
    return val;
}

void ops::render_fwd::xla(
    cudaStream_t stream,
    void **buffers,
    const char *opaque,
    std::size_t opaque_len
) {
    const auto &d = *unpack_descriptor<Descriptor>(opaque, opaque_len);

    const int B = d.batch_size;
    const int N = d.num_points;
    const int W = d.img_shape.x;
    const int H = d.img_shape.y;
    const unsigned bw = d.block_width;

    const int tiles_x = (W + bw - 1) / bw;
    const int tiles_y = (H + bw - 1) / bw;
    const int num_tiles = tiles_x * tiles_y;
    const dim3 tile_bounds = {(unsigned)tiles_x, (unsigned)tiles_y, 1};

    // ---- Unpack I/O buffers ----
    std::size_t bi = 0;
    const float3 *means3d   = static_cast<const float3 *>(buffers[bi++]);
    const float3 *scales    = static_cast<const float3 *>(buffers[bi++]);
    const float4 *quats     = static_cast<const float4 *>(buffers[bi++]);
    const float3 *colors    = static_cast<const float3 *>(buffers[bi++]);
    const float *opacities  = static_cast<const float *>(buffers[bi++]);
    const float *viewmats   = static_cast<const float *>(buffers[bi++]);
    const float3 *background = static_cast<const float3 *>(buffers[bi++]);
    float3 *out_img         = static_cast<float3 *>(buffers[bi++]);

    CUDA_THROW_IF_ERR(cudaMemsetAsync(out_img, 0, sizeof(float3) * B * H * W, stream));

    constexpr int TPB = 256;
    const int gauss_blocks = (N + TPB - 1) / TPB;
    dim3 grid_proj(B, gauss_blocks);

    // ================================================================
    // Step 1: Count visible Gaussians per camera → cam_counts[B]
    // ================================================================
    int *cam_counts;
    CUDA_THROW_IF_ERR(cudaMallocAsync(&cam_counts, sizeof(int) * B, stream));
    CUDA_THROW_IF_ERR(cudaMemsetAsync(cam_counts, 0, sizeof(int) * B, stream));

    kernels::count_visible<<<grid_proj, TPB, 0, stream>>>(
        B, N, means3d, scales, d.glob_scale, quats, viewmats,
        d.intrins, d.img_shape, tile_bounds, bw, d.clip_thresh,
        cam_counts);
    CUDA_THROW_IF_ERR(cudaGetLastError());

    // ================================================================
    // Step 2: ExclusiveSum(cam_counts) → cam_offsets[B]; total_visible
    // ================================================================
    int *cam_offsets;
    CUDA_THROW_IF_ERR(cudaMallocAsync(&cam_offsets, sizeof(int) * B, stream));
    cub_exclusive_sum(stream, cam_counts, cam_offsets, B);

    int total_visible = read_device_int(stream, cam_offsets + B - 1)
                      + read_device_int(stream, cam_counts + B - 1);

    if (total_visible == 0) {
        cudaFreeAsync(cam_counts, stream);
        cudaFreeAsync(cam_offsets, stream);
        return;
    }

    // ================================================================
    // Step 3: Project and pack → packed arrays[total_visible]
    // ================================================================
    float2 *packed_xys;
    float  *packed_depths;
    float3 *packed_conics;
    int    *packed_num_tiles;
    int    *packed_radii;
    float  *packed_colors;
    float  *packed_opacities;
    int    *cam_write_idx;

    CUDA_THROW_IF_ERR(cudaMallocAsync(&packed_xys,       sizeof(float2) * total_visible, stream));
    CUDA_THROW_IF_ERR(cudaMallocAsync(&packed_depths,    sizeof(float)  * total_visible, stream));
    CUDA_THROW_IF_ERR(cudaMallocAsync(&packed_conics,    sizeof(float3) * total_visible, stream));
    CUDA_THROW_IF_ERR(cudaMallocAsync(&packed_num_tiles, sizeof(int)    * total_visible, stream));
    CUDA_THROW_IF_ERR(cudaMallocAsync(&packed_radii,     sizeof(int)    * total_visible, stream));
    CUDA_THROW_IF_ERR(cudaMallocAsync(&packed_colors,    sizeof(float)  * total_visible * 3, stream));
    CUDA_THROW_IF_ERR(cudaMallocAsync(&packed_opacities, sizeof(float)  * total_visible, stream));
    CUDA_THROW_IF_ERR(cudaMallocAsync(&cam_write_idx,    sizeof(int)    * B, stream));

    CUDA_THROW_IF_ERR(cudaMemsetAsync(cam_write_idx,    0, sizeof(int) * B, stream));
    CUDA_THROW_IF_ERR(cudaMemsetAsync(packed_num_tiles, 0, sizeof(int) * total_visible, stream));

    kernels::project_and_pack<<<grid_proj, TPB, 0, stream>>>(
        B, N, means3d, scales, d.glob_scale, quats, viewmats,
        d.intrins, d.img_shape, tile_bounds, bw, d.clip_thresh,
        cam_offsets, cam_write_idx,
        packed_xys, packed_depths, packed_conics, packed_num_tiles, packed_radii,
        packed_colors, packed_opacities,
        colors, opacities);
    CUDA_THROW_IF_ERR(cudaGetLastError());

    cudaFreeAsync(cam_write_idx, stream);

    // ================================================================
    // Step 4: InclusiveSum(packed_num_tiles) → cum_tiles; total_intersections
    // ================================================================
    int *cum_tiles;
    CUDA_THROW_IF_ERR(cudaMallocAsync(&cum_tiles, sizeof(int) * total_visible, stream));
    cub_inclusive_sum(stream, packed_num_tiles, cum_tiles, total_visible);

    int total_intersections = read_device_int(stream, cum_tiles + total_visible - 1);

    cudaFreeAsync(packed_num_tiles, stream);

    if (total_intersections == 0) {
        cudaFreeAsync(packed_xys, stream);
        cudaFreeAsync(packed_depths, stream);
        cudaFreeAsync(packed_conics, stream);
        cudaFreeAsync(packed_radii, stream);
        cudaFreeAsync(packed_colors, stream);
        cudaFreeAsync(packed_opacities, stream);
        cudaFreeAsync(cum_tiles, stream);
        cudaFreeAsync(cam_counts, stream);
        cudaFreeAsync(cam_offsets, stream);
        return;
    }

    // ================================================================
    // Step 5: Expand packed Gaussians → tile intersections
    //         Keys encode camera: ((cam_id * num_tiles + tile_id) << 32) | depth
    // ================================================================
    int64_t *isect_ids;
    int     *flatten_ids;
    CUDA_THROW_IF_ERR(cudaMallocAsync(&isect_ids,   sizeof(int64_t) * total_intersections, stream));
    CUDA_THROW_IF_ERR(cudaMallocAsync(&flatten_ids,  sizeof(int)     * total_intersections, stream));

    {
        int blocks = (total_visible + TPB - 1) / TPB;
        kernels::expand_intersections<<<blocks, TPB, 0, stream>>>(
            total_visible, B, num_tiles,
            packed_xys, packed_depths, packed_radii, cum_tiles,
            tile_bounds, bw,
            cam_offsets, cam_counts,
            isect_ids, flatten_ids);
        CUDA_THROW_IF_ERR(cudaGetLastError());
    }

    cudaFreeAsync(cum_tiles, stream);
    cudaFreeAsync(packed_depths, stream);
    cudaFreeAsync(packed_radii, stream);

    // ================================================================
    // Step 6: Global RadixSort on isect_ids / flatten_ids
    //         Upper 32 bits = cam_id * num_tiles + local_tile_id → ensures
    //         per-camera then per-tile then per-depth ordering.
    // ================================================================
    int64_t *isect_ids_sorted;
    int     *flatten_ids_sorted;
    CUDA_THROW_IF_ERR(cudaMallocAsync(&isect_ids_sorted,   sizeof(int64_t) * total_intersections, stream));
    CUDA_THROW_IF_ERR(cudaMallocAsync(&flatten_ids_sorted,  sizeof(int)     * total_intersections, stream));

    {
        int max_global_tile = B * num_tiles;
        int msb = 32 - __builtin_clz(max_global_tile) + 1;
        cub_radix_sort_pairs(stream,
                             isect_ids, isect_ids_sorted,
                             flatten_ids, flatten_ids_sorted,
                             total_intersections, 32 + msb);
    }

    cudaFreeAsync(isect_ids, stream);
    cudaFreeAsync(flatten_ids, stream);

    // ================================================================
    // Step 7: Compute tile bin edges → tile_bins[B * num_tiles]
    // ================================================================
    int total_tile_bins = B * num_tiles;
    int2 *tile_bins;
    CUDA_THROW_IF_ERR(cudaMallocAsync(&tile_bins, sizeof(int2) * total_tile_bins, stream));
    CUDA_THROW_IF_ERR(cudaMemsetAsync(tile_bins, 0, sizeof(int2) * total_tile_bins, stream));

    {
        int blocks = (total_intersections + TPB - 1) / TPB;
        kernels::get_tile_bin_edges<<<blocks, TPB, 0, stream>>>(
            total_intersections, isect_ids_sorted, tile_bins);
        CUDA_THROW_IF_ERR(cudaGetLastError());
    }

    cudaFreeAsync(isect_ids_sorted, stream);

    // ================================================================
    // Step 8: Batched rasterization → out_img[B, H, W, 3]
    // ================================================================
    {
        float3 bg_host;
        cudaStreamSynchronize(stream);
        CUDA_THROW_IF_ERR(cudaMemcpy(&bg_host, background, sizeof(float3), cudaMemcpyDeviceToHost));

        dim3 grid_rast(B, tiles_x, tiles_y);
        dim3 block_rast(bw, bw, 1);

        kernels::batched_rasterize_fwd<<<grid_rast, block_rast, 0, stream>>>(
            B, tile_bounds, d.img_shape,
            flatten_ids_sorted, tile_bins,
            packed_xys, packed_conics,
            packed_colors, packed_opacities,
            bg_host, out_img);
        CUDA_THROW_IF_ERR(cudaGetLastError());
    }

    // ================================================================
    // Cleanup
    // ================================================================
    cudaFreeAsync(flatten_ids_sorted, stream);
    cudaFreeAsync(tile_bins, stream);
    cudaFreeAsync(packed_xys, stream);
    cudaFreeAsync(packed_conics, stream);
    cudaFreeAsync(packed_colors, stream);
    cudaFreeAsync(packed_opacities, stream);
    cudaFreeAsync(cam_counts, stream);
    cudaFreeAsync(cam_offsets, stream);

    cudaStreamSynchronize(stream);
}
