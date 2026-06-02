#pragma once

#include "common.h"
#include <cuda_runtime.h>

namespace helpers {

inline __device__ void get_bbox(const float2 center, const float2 dims,
                                const dim3 img_size, uint2 &bb_min,
                                uint2 &bb_max) {
    bb_min.x = min(max(0, (int)(center.x - dims.x)), (int)img_size.x);
    bb_max.x = min(max(0, (int)(center.x + dims.x + 1)), (int)img_size.x);
    bb_min.y = min(max(0, (int)(center.y - dims.y)), (int)img_size.y);
    bb_max.y = min(max(0, (int)(center.y + dims.y + 1)), (int)img_size.y);
}

inline __device__ void get_tile_bbox(const float2 pix_center,
                                     const float pix_radius,
                                     const dim3 tile_bounds, uint2 &tile_min,
                                     uint2 &tile_max, const int block_size) {
    float2 tile_center = {pix_center.x / (float)block_size,
                          pix_center.y / (float)block_size};
    float2 tile_radius = {pix_radius / (float)block_size,
                          pix_radius / (float)block_size};
    get_bbox(tile_center, tile_radius, tile_bounds, tile_min, tile_max);
}

inline __device__ bool compute_cov2d_bounds(const float3 cov2d, float3 &conic,
                                            float &radius) {
    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (det == 0.f)
        return false;
    float inv_det = 1.f / det;

    conic.x = cov2d.z * inv_det;
    conic.y = -cov2d.y * inv_det;
    conic.z = cov2d.x * inv_det;

    float b = 0.5f * (cov2d.x + cov2d.z);
    float v1 = b + sqrt(max(0.1f, b * b - det));
    float v2 = b - sqrt(max(0.1f, b * b - det));
    radius = ceil(3.f * sqrt(max(v1, v2)));
    return true;
}

inline __device__ float3 transform_4x3(const float *mat, const float3 p) {
    float3 out = {
        mat[0] * p.x + mat[1] * p.y + mat[2] * p.z + mat[3],
        mat[4] * p.x + mat[5] * p.y + mat[6] * p.z + mat[7],
        mat[8] * p.x + mat[9] * p.y + mat[10] * p.z + mat[11],
    };
    return out;
}

inline __device__ float2 project_pix(const float2 fxfy, const float3 p_view,
                                     const float2 pp) {
    float rw = 1.f / (p_view.z + 1e-6f);
    float2 p_proj = {p_view.x * rw, p_view.y * rw};
    float2 p_pix = {p_proj.x * fxfy.x + pp.x, p_proj.y * fxfy.y + pp.y};
    return p_pix;
}

inline __device__ mat3<float> quat_to_rotmat(const float4 quat) {
    float w = quat.x, x = quat.y, y = quat.z, z = quat.w;
    return mat3<float>(
        1.f - 2.f*(y*y + z*z), 2.f*(x*y + w*z), 2.f*(x*z - w*y),
        2.f*(x*y - w*z), 1.f - 2.f*(x*x + z*z), 2.f*(y*z + w*x),
        2.f*(x*z + w*y), 2.f*(y*z - w*x), 1.f - 2.f*(x*x + y*y)
    );
}

inline __device__ mat3<float> scale_to_mat(const float3 scale,
                                           const float glob_scale) {
    mat3<float> S = mat3<float>(1.f);
    S[0][0] = glob_scale * scale.x;
    S[1][1] = glob_scale * scale.y;
    S[2][2] = glob_scale * scale.z;
    return S;
}

inline __device__ bool clip_near_plane(const float3 p, const float *viewmat,
                                       float3 &p_view, float thresh) {
    p_view = transform_4x3(viewmat, p);
    return (p_view.z <= thresh);
}

inline __device__ void scale_rot_to_cov3d(const float3 scale,
                                          const float glob_scale,
                                          const float4 quat,
                                          float *cov3d) {
    mat3<float> R = quat_to_rotmat(quat);
    mat3<float> S = scale_to_mat(scale, glob_scale);
    mat3<float> M = R * S;
    mat3<float> tmp = M * M.transpose();
    cov3d[0] = tmp[0][0]; cov3d[1] = tmp[0][1]; cov3d[2] = tmp[0][2];
    cov3d[3] = tmp[1][1]; cov3d[4] = tmp[1][2]; cov3d[5] = tmp[2][2];
}

inline __device__ void project_cov3d_ewa(
    const float3 &mean3d, const float *cov3d, const float *viewmat,
    const float fx, const float fy, const float tan_fovx, const float tan_fovy,
    float3 &cov2d, float &compensation) {

    mat3<float> W = mat3<float>(
        viewmat[0], viewmat[4], viewmat[8],
        viewmat[1], viewmat[5], viewmat[9],
        viewmat[2], viewmat[6], viewmat[10]
    );
    float3 p = {viewmat[3], viewmat[7], viewmat[11]};
    float3 t = {
        W[0][0]*mean3d.x + W[1][0]*mean3d.y + W[2][0]*mean3d.z + p.x,
        W[0][1]*mean3d.x + W[1][1]*mean3d.y + W[2][1]*mean3d.z + p.y,
        W[0][2]*mean3d.x + W[1][2]*mean3d.y + W[2][2]*mean3d.z + p.z,
    };

    float lim_x = 1.3f * tan_fovx;
    float lim_y = 1.3f * tan_fovy;
    t.x = t.z * min(lim_x, max(-lim_x, t.x / t.z));
    t.y = t.z * min(lim_y, max(-lim_y, t.y / t.z));

    float rz = 1.f / t.z;
    float rz2 = rz * rz;

    mat3<float> J = mat3<float>(
        fx*rz, 0.f, 0.f,
        0.f, fy*rz, 0.f,
        -fx*t.x*rz2, -fy*t.y*rz2, 0.f
    );
    mat3<float> T = J * W;

    mat3<float> V = mat3<float>(
        cov3d[0], cov3d[1], cov3d[2],
        cov3d[1], cov3d[3], cov3d[4],
        cov3d[2], cov3d[4], cov3d[5]
    );

    mat3<float> cov = T * V * T.transpose();

    float c00 = cov[0][0], c11 = cov[1][1], c01 = cov[0][1];
    float det_orig = c00 * c11 - c01 * c01;
    cov2d.x = c00 + 0.3f;
    cov2d.y = c01;
    cov2d.z = c11 + 0.3f;
    float det_blur = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    compensation = sqrt(max(0.f, det_orig / det_blur));
}

} // namespace helpers
