#define GET_PT_BFYX(pts, b, n, c) pts[((b) * INPUT0_DIMS[1] + (n)) * INPUT0_DIMS[2] + (c)]

__kernel void ov_furthest_point_sampling(
    __global const float* pts,      // (B, N, 3)
    __global const int* npoint_ptr, // npoint scalar
    __global int* output,           // (B, npoint)
    __global float* dist            // (B, N)
) {
    int tid = get_global_id(0);
    int b   = get_global_id(1);  // batch index

    int B = INPUT0_DIMS[0];  // batch size
    int N = INPUT0_DIMS[1];  // number of points
    int C = INPUT0_DIMS[2];  // channels (should be 3)
    int npoint = INPUT1_DIMS[0];

    if (b >= B) return;

    // Offsets for batch memory
    int pts_offset  = b * N * C;
    int dist_offset = b * N;
    int out_offset  = b * npoint;

    int local_id = get_local_id(0);
    int local_size = get_local_size(0);

    // Initialize dist array per batch
    for (int i = local_id; i < N; i += local_size)
        dist[dist_offset + i] = FLT_MAX;

    barrier(CLK_LOCAL_MEM_FENCE);

    // Initialize first output point = 0
    if (local_id == 0) output[out_offset] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    __local float local_dist[256];  // adjust if local_size changes
    __local int local_idx[256];

    for (int j = 1; j < npoint; ++j) {
        int last_idx = output[out_offset + j - 1];

        float last_x = pts[pts_offset + last_idx * 3 + 0];
        float last_y = pts[pts_offset + last_idx * 3 + 1];
        float last_z = pts[pts_offset + last_idx * 3 + 2];

        // update dist array in parallel
        for (int i = local_id; i < N; i += local_size) {
            float dx = pts[pts_offset + i * 3 + 0] - last_x;
            float dy = pts[pts_offset + i * 3 + 1] - last_y;
            float dz = pts[pts_offset + i * 3 + 2] - last_z;
            float d = dx*dx + dy*dy + dz*dz;
            if (d < dist[dist_offset + i])
                dist[dist_offset + i] = d;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // find max dist and argmax with local reduction
        float best_dist = -1.0f;
        int best_idx = -1;
        for (int i = local_id; i < N; i += local_size) {
            float v = dist[dist_offset + i];
            if (v > best_dist) {
                best_dist = v;
                best_idx = i;
            }
        }

        local_dist[local_id] = best_dist;
        local_idx[local_id] = best_idx;
        barrier(CLK_LOCAL_MEM_FENCE);

        // reduce within local memory to find global max
        for (int stride = local_size/2; stride > 0; stride /= 2) {
            if (local_id < stride) {
                if (local_dist[local_id] < local_dist[local_id + stride]) {
                    local_dist[local_id] = local_dist[local_id + stride];
                    local_idx[local_id] = local_idx[local_id + stride];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (local_id == 0) output[out_offset + j] = local_idx[0];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
