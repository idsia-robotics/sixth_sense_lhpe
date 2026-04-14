import torch
from lidar_human_pose_estimation.utils import sensor_utils, geom_utils
from lidar_human_pose_estimation.core.config import _eps


def circular_dilation(input_tensor: torch.Tensor, dilation_pixels: torch.Tensor) -> torch.Tensor:
    """
    Applies a circular max pooling operation along the N_rays dimension for all timestamps using PyTorch.
    This version uses a max pooling kernel along the final dimension (rays) with circular padding.

    Parameters:
        input_tensor (torch.Tensor): A 2D tensor of shape (N_Timestamps, N_rays).
        dilation_pixels (int): Number of pixels to dilate on each side of a nonzero element.

    Returns:
        torch.Tensor: A tensor with circular max pooling applied.
    """
    if not isinstance(input_tensor, torch.Tensor):
        raise ValueError("Input must be a torch.Tensor.")
    if len(input_tensor.shape) != 2:
        raise ValueError("Input tensor must be 2D (N_Timestamps x N_rays).")
    if not isinstance(dilation_pixels, int) or dilation_pixels < 0:
        raise ValueError("dilation_pixels must be a non-negative integer.")

    # Get the shape of the tensor
    num_timestamps, num_rays = input_tensor.shape

    # Create a kernel size to apply max pooling
    kernel_size = 1 + 2 * dilation_pixels
    stride = 1  # We want to slide the window across each element (no skipping)

    # Circular padding by repeating the tensor along the rays dimension
    padded_tensor = torch.cat([input_tensor, input_tensor, input_tensor], dim=1)

    # Apply max pooling along the rays dimension using a sliding window
    dilated_tensor = padded_tensor.unfold(1, kernel_size, stride)

    # Now we can take the max along the unfolded dimension (the window dimension)
    dilated_tensor = dilated_tensor.min(dim=2)[0]  # Max along the 'kernel_size' dimension

    # Extract the central part corresponding to the original array (undo the 3x padding)
    dilated_tensor = dilated_tensor[:, num_rays - dilation_pixels : 2 * num_rays - dilation_pixels]

    return dilated_tensor


def circular_erosion(input_tensor: torch.Tensor) -> torch.Tensor:
    n_timestamps, n_rays = input_tensor.shape

    # Circular shift tensors to compare neighbors
    left_shifted = torch.roll(input_tensor, shifts=1, dims=1)
    right_shifted = torch.roll(input_tensor, shifts=-1, dims=1)

    # A ray is marked only if it is 1 and its neighbors (left and right) are 0
    eroded_tensor = torch.logical_and(torch.logical_and(input_tensor, left_shifted), right_shifted)

    return eroded_tensor.float()


def relative_bearing(absolute_bearing: torch.Tensor, rays_angles: torch.Tensor, default_value: float) -> torch.Tensor:
    valid_mask = absolute_bearing != default_value

    # Zero relative bearing is the direction of the sensor, so people looking at the robot
    relative_bearing = (absolute_bearing - torch.pi) - rays_angles.unsqueeze(0)  # Shape: [N_timestamps, N_rays]

    relative_bearing = (relative_bearing + torch.pi) % (2 * torch.pi) - torch.pi

    # Apply the mask to ignore invalid entries
    relative_bearing[~valid_mask] = default_value

    return relative_bearing


def absolute_bearing(relative_bearing: torch.Tensor, rays_angles: torch.Tensor) -> torch.Tensor:
    absolute_bearing = (relative_bearing + rays_angles.unsqueeze(0)) + torch.pi  # Shape: [N_timestamps, N_rays]

    absolute_bearing = (absolute_bearing + torch.pi) % (2 * torch.pi) - torch.pi

    return absolute_bearing


def gt_from_sensor_detections(
    user_poses_sensor_frame: torch.Tensor,
    detection_sensor_poses_gt_frame: torch.Tensor,
    detection_sensor_info: dict,
    gt_sensor_info: dict,
):
    default_value = 1 / _eps

    # 1) Mask to represent detection sensor field of view
    # Treat sensor like a lidar with point at max range to get the field of view mask
    angles_detection_sensor_fov = sensor_utils.get_sensor_angles(detection_sensor_info)
    polar_coords_detection_sensor_fov = torch.stack(
        [
            torch.ones(detection_sensor_poses_gt_frame.size(0), angles_detection_sensor_fov.size(0))
            * detection_sensor_info["range_max"],
            angles_detection_sensor_fov.tile(detection_sensor_poses_gt_frame.size(0), 1),
        ],
        dim=-1,
    )
    # Polar coordinates in robot frame
    polar_coords_detection_sensor_fov = geom_utils.transform_polar(
        polar_coords_detection_sensor_fov, detection_sensor_poses_gt_frame.unsqueeze(1), alignment_axis="y"
    )
    detection_sensor_fov_to_gt_sensor = geom_utils.aggregate_scan(
        polar_coords_detection_sensor_fov[..., 0],
        polar_coords_detection_sensor_fov[..., 1],
        sensor_utils.get_sensor_angles(gt_sensor_info),
        wrap=True,
        reduce="amin",
        default=default_value,
    )
    # Binarize the mask
    detection_sensor_mask = torch.where(detection_sensor_fov_to_gt_sensor == default_value, 0, 1)

    # Body tracking data preprocessing
    user_poses_gt_frame = torch.matmul(detection_sensor_poses_gt_frame.unsqueeze(1), user_poses_sensor_frame)

    # Sensor data has y axis as forward direction, so we need to rotate the poses
    R = torch.tensor(
        [
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=torch.float64,
    )
    user_poses_gt_frame[:, :, 0:3, 0:3] = torch.matmul(user_poses_gt_frame[:, :, 0:3, 0:3], R)

    # 2) Value to represent the distance
    # 3) Value to represent the presence of a human
    # 4) Value to represent the relative orientation of the human
    users_distance_to_gt_sensor, users_presence_to_gt_sensor, user_relative_orientations_to_gt_sensor = (
        gt_from_gt_frame_detections(user_poses_gt_frame, gt_sensor_info)
    )

    return (
        detection_sensor_mask,
        users_distance_to_gt_sensor,
        users_presence_to_gt_sensor,
        user_relative_orientations_to_gt_sensor,
    )


def gt_from_gt_frame_detections(user_poses_gt_frame: torch.Tensor, gt_sensor_info: dict):
    default_value = 1 / _eps
    # Vector of angles for the gt_sensor
    gt_sensor_angles = sensor_utils.get_sensor_angles(gt_sensor_info)

    ##### Value to represent the distance #####
    users_positions_gt_frame_cartesian, users_yaws_gt_frame = geom_utils.extract_cartesian_coordinates_and_yaw(
        user_poses_gt_frame
    )

    users_positions_gt_frame_polar = geom_utils.cartesian_to_polar(users_positions_gt_frame_cartesian.view(-1, 2)).view(
        users_positions_gt_frame_cartesian.size(0), users_positions_gt_frame_cartesian.size(1), 2
    )
    users_distance_to_gt_sensor = geom_utils.aggregate_scan(
        users_positions_gt_frame_polar[..., 0],
        users_positions_gt_frame_polar[..., 1],
        sensor_utils.get_sensor_angles(gt_sensor_info),
        wrap=True,
        reduce="amin",
        default=default_value,
    )

    users_distance_to_gt_sensor = circular_dilation(users_distance_to_gt_sensor, 1)
    users_distance_to_gt_sensor = torch.where(
        users_distance_to_gt_sensor == default_value, 0, users_distance_to_gt_sensor
    )

    ##### Value to represent the presence of a human #####
    users_presence_to_gt_sensor = torch.where(users_distance_to_gt_sensor == 0, 0, 1)

    ##### Value to represent the relative orientation of the human #####
    body_tracking_yaw_polar_coords = torch.cat(
        [users_yaws_gt_frame, users_positions_gt_frame_polar[..., 1].unsqueeze(dim=-1)], dim=-1
    )
    user_absolute_orientations_to_gt_sensor = geom_utils.aggregate_scan(
        body_tracking_yaw_polar_coords[..., 0],
        body_tracking_yaw_polar_coords[..., 1],
        sensor_utils.get_sensor_angles(gt_sensor_info),
        wrap=True,
        reduce="amin",
        default=default_value,
    )

    users_relative_orientations_to_gt_sensor = relative_bearing(
        user_absolute_orientations_to_gt_sensor, sensor_utils.get_sensor_angles(gt_sensor_info), default_value
    )
    users_relative_orientations_to_gt_sensor = circular_dilation(users_relative_orientations_to_gt_sensor, 1)
    users_relative_orientations_to_gt_sensor[users_relative_orientations_to_gt_sensor == default_value] = 0

    return users_distance_to_gt_sensor, users_presence_to_gt_sensor, users_relative_orientations_to_gt_sensor
