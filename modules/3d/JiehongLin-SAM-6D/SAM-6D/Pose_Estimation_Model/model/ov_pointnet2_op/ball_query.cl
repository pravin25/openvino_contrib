// ball_query.cl
// Fixed to prevent early exit and race conditions

__kernel void ov_ball_query(
    __global const INPUT0_TYPE* new_xyz,   // (B, npoint, 3)
    __global const INPUT1_TYPE* xyz,       // (B, N, 3)
    __global OUTPUT0_TYPE* idx)            // (B, npoint, nsample)
{
    uint batch_index = get_global_id(0);      
    uint point_index = get_global_id(1);      

    int b = INPUT0_DIMS[0];
    int n = INPUT1_DIMS[1];
    int npoint = INPUT0_DIMS[1];
    // nsample and radius are passed as #defines from XML

    if (batch_index >= b || point_index >= npoint) return;

    float radius2 = radius * radius;
    int cnt = 0;

    // Offsets
    uint new_xyz_offset = batch_index * npoint * 3 + point_index * 3;
    uint xyz_batch_offset = batch_index * n * 3;
    uint output_offset = batch_index * npoint * nsample + point_index * nsample;

    INPUT0_TYPE new_x = new_xyz[new_xyz_offset + 0];
    INPUT0_TYPE new_y = new_xyz[new_xyz_offset + 1];
    INPUT0_TYPE new_z = new_xyz[new_xyz_offset + 2];

    // Stream directly from global memory to avoid local memory overflow and barrier deadlocks
    for (int k = 0; k < n && cnt < nsample; ++k) {
        INPUT1_TYPE x = xyz[xyz_batch_offset + k * 3 + 0];
        INPUT1_TYPE y = xyz[xyz_batch_offset + k * 3 + 1];
        INPUT1_TYPE z = xyz[xyz_batch_offset + k * 3 + 2];

        float d2 = (float)(new_x - x) * (new_x - x) +
                   (float)(new_y - y) * (new_y - y) +
                   (float)(new_z - z) * (new_z - z);

        if (d2 < radius2) {
            if (cnt == 0) {
                // Initialize all slots with the first neighbor index
                for (int l = 0; l < nsample; ++l) {
                    idx[output_offset + l] = k;
                }
            }
            // Assign current neighbor
            idx[output_offset + cnt] = k;
            cnt++;
        }
    }
}
