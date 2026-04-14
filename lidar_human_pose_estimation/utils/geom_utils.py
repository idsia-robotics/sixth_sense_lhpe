import torch
from lidar_human_pose_estimation.core.config import _eps
from typing import Tuple
from numbers import Number
from munkres import Munkres



def polar_to_cartesian(x: torch.Tensor) -> torch.Tensor:
    """Converts a tensor of polar coordinates to cartesian coordinates."""
    return x[..., 0, None] * torch.stack(
        [torch.cos(x[..., 1]), torch.sin(x[..., 1])], dim=-1
    )


def cartesian_to_polar(x: torch.Tensor) -> torch.Tensor:
    """Converts a tensor of cartesian coordinates to polar coordinates."""
    return torch.stack(
        [torch.linalg.norm(x, dim=-1), torch.atan2(x[..., 1], x[..., 0])], dim=-1
    )


def transform_polar(
    x: torch.Tensor, pose: torch.Tensor, alignment_axis: str
) -> torch.Tensor:
    """Applies a transformation to a tensor of polar coordinates."""
    cartesian_coords = polar_to_cartesian(x)
    if alignment_axis == "z":
        coords_homo = torch.cat(
            [
                cartesian_coords,
                torch.zeros_like(cartesian_coords[..., :1]),
                torch.ones_like(cartesian_coords[..., :1]),
            ],
            dim=-1,
        )
    elif alignment_axis == "y":
        coords_homo = torch.cat(
            [
                -cartesian_coords[..., 1].unsqueeze(dim=-1),
                torch.zeros_like(cartesian_coords[..., :1]),
                cartesian_coords[..., 0].unsqueeze(dim=-1),
                torch.ones_like(cartesian_coords[..., :1]),
            ],
            dim=-1,
        )
    else:
        raise ValueError(f'Alignment axis "{alignment_axis}" not supported.')

    transformed_cartesian_coords = torch.matmul(pose, coords_homo[..., None])[
        ..., :2, 0
    ]
    transformed_polar_coords = cartesian_to_polar(transformed_cartesian_coords)
    return transformed_polar_coords


def invert_transforms(transforms: torch.Tensor) -> torch.Tensor:
    # Create a mask for matrices with all NaN values
    not_nan_mask = ~transforms.isnan().view(-1, 16).all(dim=-1)

    # Replace NaN matrices with identity matrices
    tensor_no_nan = transforms.clone()[not_nan_mask]

    # Perform batch inversion
    inverted_matrices_no_nan = torch.linalg.inv(tensor_no_nan)

    inverted_matrices = torch.full(
        transforms.shape, float("nan"), device=transforms.device, dtype=transforms.dtype
    )
    inverted_matrices[not_nan_mask] = inverted_matrices_no_nan

    return inverted_matrices


def compose_transform(
    transform_a_to_b: torch.Tensor,
    transform_b_to_c: torch.Tensor,
) -> torch.Tensor:

    transform_a_to_c = torch.matmul(transform_b_to_c, transform_a_to_b)
    return transform_a_to_c


def aggregate_scan_history(
    scan: torch.Tensor,
    scan_angle: torch.Tensor,
    out_angle: torch.Tensor,
    default: Number = 1e10,
    reduce: str = "amin",
    wrap: bool = False,
) -> torch.Tensor:
    num_timestamps = scan.size(0)
    history_length = scan.size(1)
    scan_virtual = aggregate_scan(
        scan.view(-1, scan.size(-1)),
        scan_angle.view(-1, scan.size(-1)),
        out_angle,
        default,
        reduce,
        wrap,
    )
    return scan_virtual.view(num_timestamps, history_length, -1)


def aggregate_scan(
    scan: torch.Tensor,
    scan_angle: torch.Tensor,
    out_angle: torch.Tensor,
    default: Number = 1e10,
    reduce: str = "amin",
    wrap: bool = False,
) -> torch.Tensor:
    """Aggregates scan readings by dividing into out_angle buckets and applying reduce function, excluding NaNs.

    Args:
        scan: scan readings tensor (must have same shape as scan_angle).
        scan_angle: scan readings angle tensor (must have same shape as scan).
        out_angle: new angles used to discretize scan into buckets.
        default: value used for buckets with no points lying inside, must be consistent with reduce (e.g., a large value when reduce is amin).
        reduce: aggregation function (see: https://pytorch.org/docs/stable/generated/torch.Tensor.index_reduce_.html#torch-tensor-index-reduce).
        wrap: when set fused first and last buckets (useful when scan_angle covers 360 degrees).

    Returns:
        The aggregated scan.
    """
    length = scan.size(0)
    num_rays = out_angle.size(0)

    # Offset to handle bucketing correctly
    offset = out_angle.diff().mean() / 2
    indices = torch.bucketize(scan_angle.contiguous(), out_angle + offset)

    if wrap:
        indices = indices % num_rays

    indices = torch.arange(length)[..., None] * num_rays + indices

    # Mask to filter out NaNs from scan tensor
    valid_mask = ~torch.isnan(scan)

    # Replace NaNs with a default value
    scan = torch.where(valid_mask, scan, torch.full_like(scan, default))

    return (
        torch.full((length * num_rays,), default, dtype=scan.dtype)
        .index_reduce_(
            dim=0, index=indices.flatten(), source=scan.flatten(), reduce=reduce
        )
        .view(length, num_rays)
    )


##############


def translation_matrix(transl):
    """Given a batch of positions (x, y, z) returns a batch of pure-translation 4x4 homogenous transformation matrices."""
    bs = transl.size(0)
    t = torch.eye(4, device=transl.device, dtype=transl.dtype).repeat(bs, 1, 1)
    t[:, :3, 3] = transl[:, :3]
    return t


def quaternion_matrix(quat, eps=_eps):
    """Given a batch of quaternion rotations (qx, qy, qz, qw) returns a batch of pure-rotation 4x4 homogenous transformation matrices."""
    bs = quat.size(0)
    nquat = (quat * quat).sum(dim=-1, keepdim=True)
    res = torch.eye(4, device=quat.device, dtype=quat.dtype).repeat(bs, 1, 1)
    quat = quat * torch.sqrt(2.0 / nquat)
    q = torch.bmm(quat.unsqueeze(-1), quat.unsqueeze(-2))

    row1 = torch.stack(
        [
            1.0 - q[:, 1, 1] - q[:, 2, 2],
            q[:, 0, 1] - q[:, 2, 3],
            q[:, 0, 2] + q[:, 1, 3],
        ]
    )
    row2 = torch.stack(
        [
            q[:, 0, 1] + q[:, 2, 3],
            1.0 - q[:, 0, 0] - q[:, 2, 2],
            q[:, 1, 2] - q[:, 0, 3],
        ]
    )
    row3 = torch.stack(
        [
            q[:, 0, 2] - q[:, 1, 3],
            q[:, 1, 2] + q[:, 0, 3],
            1.0 - q[:, 0, 0] - q[:, 1, 1],
        ]
    )

    magic = torch.stack([row1, row2, row3]).permute(2, 0, 1)
    res[nquat[:, 0] >= eps, :3, :3] = magic[nquat[:, 0] >= eps, ...]
    return res


def pose_to_matrix(pose):
    """Given a batch of poses (x, y, z, qx, qy, qz, qw), returns a batch of 4x4 homogenous transformation matrices, robust to NaN values."""
    if pose.dim() == 3:
        nan_mask = torch.isnan(pose).any(dim=-1)  # Detect NaN rows
        batch_size, num_poses, _ = pose.shape
        pose = pose.view(-1, 7)  # Flatten batch for processing

        # Initialize output with NaN matrices
        res = torch.full(
            (batch_size * num_poses, 4, 4),
            float("nan"),
            device=pose.device,
            dtype=pose.dtype,
        )

        # Process valid rows
        valid_pose = pose[~nan_mask.view(-1)]
        if valid_pose.numel() > 0:
            t = translation_matrix(valid_pose[:, :3])
            r = quaternion_matrix(valid_pose[:, 3:])
            valid_res = torch.bmm(t, r)
            res[~nan_mask.view(-1)] = valid_res

        return res.view(batch_size, num_poses, 4, 4)

    # Original implementation for 2D pose batch
    nan_mask = torch.isnan(pose).any(dim=-1)  # Detect NaN rows
    res = torch.full(
        (pose.size(0), 4, 4), float("nan"), device=pose.device, dtype=pose.dtype
    )

    valid_pose = pose[~nan_mask]
    if valid_pose.numel() > 0:
        t = translation_matrix(valid_pose[:, :3])
        r = quaternion_matrix(valid_pose[:, 3:])
        res[~nan_mask] = torch.bmm(t, r)

    return res


def matrix_to_pose(mat):
    """Given a batch of 4x4 homogenous transformation matrices returns a batch of poses (x, y, z, qx, qy, qz, qw)."""
    bs = mat.size(0)
    t = mat[..., :3, 3]

    # if    m00 + m11 + m22 > 0     -> c0
    # elif  m00 > m11 & m00 > m22   -> c1
    # elif  m11 > m22               -> c2
    # else                          -> c3
    c0 = mat[..., 0, 0] + mat[..., 1, 1] + mat[..., 2, 2] > 0
    c1 = (mat[..., 0, 0] > mat[..., 1, 1]) & (mat[..., 0, 0] > mat[..., 2, 2])
    c2 = mat[..., 1, 1] > mat[..., 2, 2]
    c3 = ~(c0 | c1 | c2)
    c2 = ~(c0 | c1) & c2
    c1 = ~(c0) & c1

    # a -> sign(m11) in S is negative
    # b -> sign(m22) in S is negative
    # c -> sign(m00) in S is negative
    a = (c1 | c3).to(torch.float)
    b = (c1 | c2).to(torch.float)
    c = (c2 | c3).to(torch.float)

    S = 2 * torch.sqrt(
        1
        + ((-1) ** c) * mat[..., 0, 0]
        + ((-1) ** a) * mat[..., 1, 1]
        + ((-1) ** b) * mat[..., 2, 2]
    )

    r0 = 0.25 * S * S
    r1 = mat[..., 0, 2] - ((-1) ** a) * mat[..., 2, 0]
    r2 = mat[..., 1, 0] - ((-1) ** b) * mat[..., 0, 1]
    r3 = mat[..., 2, 1] - ((-1) ** c) * mat[..., 1, 2]

    r0_index = (
        3 * c0.to(torch.long)
        + 0 * c1.to(torch.long)
        + 1 * c2.to(torch.long)
        + 2 * c3.to(torch.long)
    )
    r1_index = (r0_index + 2) % 4
    r2_index = ((r0_index - ((-1) ** a)) % 4).to(torch.long)
    r3_index = ((r0_index + ((-1) ** a)) % 4).to(torch.long)

    r = torch.zeros(bs, 4, device=mat.device, dtype=mat.dtype)
    r.index_put_((torch.arange(bs, device=r.device), r0_index), r0)
    r.index_put_((torch.arange(bs, device=r.device), r1_index), r1)
    r.index_put_((torch.arange(bs, device=r.device), r2_index), r2)
    r.index_put_((torch.arange(bs, device=r.device), r3_index), r3)

    r = r / S.unsqueeze(-1)
    r = r / torch.norm(r, dim=-1, keepdim=True)
    res = torch.cat([t, r], dim=-1)
    return res


def extract_cartesian_coordinates_and_yaw(
    transform_matrices: torch.Tensor,
) -> Tuple[torch.Tensor]:
    """
    Extracts the x, y coordinates and yaw angle from a tensor of homogeneous transformation matrices.

    Args:
        transform_matrices: Tensor of shape [N_steps, N_body, 4, 4] representing N_body homogeneous transformation matrices for each timestamp.

    Returns:
        - coordinates: Tensor of shape [N_steps, N_body, 2] representing the (x, y) coordinates.
        - yaw: Tensor of shape [N_steps, N_body, 1] representing the yaw angle (around Z axis).
    """

    poses = matrix_to_pose(transform_matrices.view(-1, 4, 4)).view(
        -1, transform_matrices.size(1), 7
    )

    # Extract x, y coordinates (3rd column of the 4x4 matrix)
    coordinates = poses[:, :, :2]  # Shape [batch_size, body_count, 2]

    # Extract the rotation part of the matrix (top-left 3x3)
    rotation_matrices = transform_matrices[:, :, :3, :3]  # Shape [batch_size, body_count, 3, 3]

    # Compute yaw from the rotation matrix x-axis (atan2 of the elements in the top row of the 3x3 rotation matrix)
    yaw = torch.atan2(rotation_matrices[:, :, 1, 0], rotation_matrices[:, :, 0, 0])  # Shape [batch_size, body_count]

    # Add an extra dimension to yaw for consistency
    yaw = yaw.unsqueeze(-1)  # Shape [batch_size, body_count, 1]

    return coordinates, yaw


def approximate_yaw_from_hips(
    hip_right: torch.Tensor, hip_left: torch.Tensor
) -> torch.Tensor:
    hip_right_pos = hip_right[:, :, :2, -1]
    hip_left_pos = hip_left[:, :, :2, -1]

    hip_right_to_hip_left = hip_left_pos - hip_right_pos
    hip_right_to_hip_left = hip_right_to_hip_left.view(-1, 2)
    # Add third dimension of all zeros to the vector
    hip_right_to_hip_left = torch.cat(
        (
            hip_right_to_hip_left,
            torch.zeros(hip_right_to_hip_left.size(0), 1, device=hip_right.device),
        ),
        dim=1,
    )

    z_axis = torch.tensor([[0, 0, 1]], dtype=hip_right_to_hip_left.dtype)
    x_axis = torch.cross(hip_right_to_hip_left, z_axis)

    yaw = torch.atan2(x_axis[:, 1], x_axis[:, 0]).unsqueeze(-1)

    return yaw.view(hip_right.size(0), hip_right.size(1), 1)


def bucketize_readings(readings, bin_count=400, bin_boundaries=[-12, 12]):

    cell_angles = torch.linspace(-torch.pi, torch.pi, readings.shape[-1])
    cell_cos = cell_angles.cos()
    cell_sin = cell_angles.sin()
    cell_cs = torch.stack([cell_cos, cell_sin], axis=0)[:, None, None, :]
    point_cloud = readings.mul(cell_cs)

    bins = torch.linspace(*bin_boundaries, bin_count)
    point_cloud_coords = torch.bucketize(point_cloud, bins) - 1

    pc_map = torch.zeros(
            (point_cloud.shape[1], point_cloud.shape[2], bins.numel(), bins.numel()),
            dtype=torch.bool,
        )

    batch_indexes = torch.repeat_interleave(
        torch.arange(point_cloud.shape[1]),
        point_cloud.shape[2] * point_cloud.shape[3],
    )

    history_indexes = torch.tile(
        torch.repeat_interleave(torch.arange(point_cloud.shape[2]), point_cloud.shape[3]),
        (point_cloud.shape[1],),
    )
    point_cloud_coords = point_cloud_coords.reshape(2, -1)
    pc_map[batch_indexes, history_indexes, point_cloud_coords[1, ...], point_cloud_coords[0, ...]] = 1
    return pc_map


def detection_matching(
        pred_positions,
        true_positions,
        distance_threshold_m = 1.5
):
    if len(true_positions) == 0:
        return true_positions, [], None
    elif len(pred_positions) == 0:
        return [], pred_positions, None
    else:
        diff = torch.cdist(
            torch.stack(true_positions)[None, ...],
            torch.stack(pred_positions)[None, ...],
        )
        m = Munkres()
        indexes = m.compute(diff[0, ...].detach().cpu().tolist())
        indexes = list(filter(lambda pair: diff[0, pair[0], pair[1]] <= distance_threshold_m, indexes))
        matched_gts = [i[0] for i in indexes]
        matched_preds = [i[1] for i in indexes]
        return matched_gts, matched_preds, diff



    