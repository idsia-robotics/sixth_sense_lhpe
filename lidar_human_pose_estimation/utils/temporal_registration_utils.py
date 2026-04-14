import h5py
import numpy as np
import torch
from typing import Tuple, List
from numbers import Number
from lidar_human_pose_estimation.core.config import _eps
from lidar_human_pose_estimation.utils import geom_utils
from lidar_human_pose_estimation.utils import sensor_utils


def generate_lidar_history(lidar_data: torch.Tensor, history_parameters: dict) -> torch.Tensor:
    """
    Generates a tensor containing the history of lidar scans for each timestamp using vectorized operations.

    Args:
        lidar_data (torch.Tensor): Input tensor of shape [N_timestamps, N_rays, N_dim].
        history_length (int): Number of historical steps to include.

    Returns:
        torch.Tensor: Output tensor of shape [N_timestamps, history_length, N_rays, N_dim].
    """
    n_timestamps = lidar_data.shape[0]

    # Create a tensor of indices that we can gather from lidar_data
    indices = torch.arange(n_timestamps).unsqueeze(1).expand(-1, history_parameters["length"]) - (
        torch.arange(history_parameters["length"]) * history_parameters["stride"]
    ).view(1, -1).expand(n_timestamps, -1)

    indices[indices < 0] = 0

    # Gather the necessary history using these indices
    history_tensor = lidar_data[indices]

    return history_tensor


def create_batches(
    total_length: int,
    max_batch_size: int,
    history_parameters: dict,
) -> List[Tuple[int, int]]:
    batches_read = []
    batches_write = []
    start = 0

    while start < total_length:
        end = min(start + max_batch_size, total_length)
        batch_start = max(0, start - history_parameters["length"] * history_parameters["stride"])
        batches_read.append((batch_start, end))
        batches_write.append((start, end))
        start += max_batch_size

    return batches_read, batches_write


def virtual_scan_time_registration(
    hdf5_file: h5py.File,
    history_parameters: dict,
    device: torch.device,
    max_batch_size: int = 1000,
) -> torch.Tensor:
    # Sensors information
    lidar_front = {
        "angle_min": hdf5_file.attrs["scan_raw_angle_min"],
        "angle_max": hdf5_file.attrs["scan_raw_angle_max"],
        "range_min": hdf5_file.attrs["scan_raw_range_min"],
        "range_max": min(hdf5_file.attrs["scan_raw_range_max"], 10.0),
        "angle_increment": hdf5_file.attrs["scan_raw_angle_increment"],
    }

    lidar_back = {
        "angle_min": hdf5_file.attrs["scan_raw_back_angle_min"],
        "angle_max": hdf5_file.attrs["scan_raw_back_angle_max"],
        "range_min": hdf5_file.attrs["scan_raw_back_range_min"],
        "range_max": min(hdf5_file.attrs["scan_raw_back_range_max"], 15.0),
        "angle_increment": hdf5_file.attrs["scan_raw_back_angle_increment"],
    }

    lidar_virtual = {
        "angle_min": hdf5_file.attrs["scan_virtual_angle_min"],
        "angle_max": hdf5_file.attrs["scan_virtual_angle_max"],
        "range_min": hdf5_file.attrs["scan_virtual_range_min"],
        "range_max": hdf5_file.attrs["scan_virtual_range_max"],
        "angle_increment": hdf5_file.attrs["scan_virtual_angle_increment"],
    }
    # File data
    data = {
        "scan_raw": hdf5_file["scan_raw"][:],
        "scan_raw_back": hdf5_file["scan_raw_back"][:],
        "body_count": hdf5_file["body_count"][:],
        "odom": hdf5_file["odom"][:, :7],
        "odom_corrected": hdf5_file["dlo_ros__odom"][:, :7] if "dlo_ros__odom" in hdf5_file else None,
        "tf_base_link_wrt_odom": hdf5_file["tf_base_link_wrt_odom"][:],
        "tf_base_laser_link_wrt_base_link": hdf5_file["tf_base_laser_link_wrt_base_link"][:],
        "tf_base_laser_back_link_wrt_base_link": hdf5_file["tf_base_laser_back_link_wrt_base_link"][:],
    }

    # READ DATA FROM FILE
    scan_front = torch.tensor(data["scan_raw"], dtype=torch.float64)
    scan_back = torch.tensor(data["scan_raw_back"], dtype=torch.float64)

    # Static TF
    laser_front_to_base_pose = torch.tensor(data["tf_base_laser_link_wrt_base_link"][0], dtype=torch.float64)
    laser_front_to_base_pose = geom_utils.pose_to_matrix(laser_front_to_base_pose.unsqueeze(dim=0))
    laser_back_to_base_pose = torch.tensor(data["tf_base_laser_back_link_wrt_base_link"][0], dtype=torch.float64)
    laser_back_to_base_pose = geom_utils.pose_to_matrix(laser_back_to_base_pose.unsqueeze(dim=0))

    # Dynamic TF
    if data["odom_corrected"] is not None:
        base_to_odom = torch.tensor(data["odom_corrected"], dtype=torch.float64)
    else:
        base_to_odom = torch.tensor(data["odom"], dtype=torch.float64)
    base_to_odom = geom_utils.pose_to_matrix(base_to_odom)
    odom_to_base = geom_utils.invert_transforms(base_to_odom).unsqueeze(1)

    # METADATA PROCESSING
    # From lasers frames to odom common frame
    laser_front_to_odom = geom_utils.compose_transform(
        transform_a_to_b=laser_front_to_base_pose,
        transform_b_to_c=base_to_odom,
    ).unsqueeze(1)
    laser_back_to_odom = geom_utils.compose_transform(
        transform_a_to_b=laser_back_to_base_pose,
        transform_b_to_c=base_to_odom,
    ).unsqueeze(1)

    # Vector of angles for the virtual scan
    angles_virtual = sensor_utils.get_sensor_angles(lidar_virtual)

    # Create batches
    batches_read, batches_write = create_batches(scan_front.size(0), max_batch_size, history_parameters)

    scan_virtual_history = torch.empty(
        scan_front.size(0), history_parameters["length"], angles_virtual.size(0), device=device
    )

    # Polar coordinates of the front scan [rho, theta, 1]
    scan_front_polar_coords_lidar = torch.stack(
        [scan_front, sensor_utils.get_sensor_angles(lidar_front).tile(scan_front.size(0), 1)],
        dim=-1,
    )
    del scan_front
    scan_back_polar_coords_lidar = torch.stack(
        [scan_back, sensor_utils.get_sensor_angles(lidar_back).tile(scan_back.size(0), 1)],
        dim=-1,
    )
    del scan_back

    scan_front_polar_coords_odom = geom_utils.transform_polar(
        scan_front_polar_coords_lidar, laser_front_to_odom, alignment_axis="z"
    )
    del scan_front_polar_coords_lidar
    scan_back_polar_coords_odom = geom_utils.transform_polar(
        scan_back_polar_coords_lidar, laser_back_to_odom, alignment_axis="z"
    )
    del scan_back_polar_coords_lidar

    # DATA PROCESSING IN BATCHES
    for batch_idx in range(len(batches_read)):
        batch_read_start = batches_read[batch_idx][0]
        batch_read_end = batches_read[batch_idx][1]
        batch_write_start = batches_write[batch_idx][0]
        batch_write_end = batches_write[batch_idx][1]
        batch_size = batch_read_end - batch_read_start

        # Divide the scan into history
        scan_front_polar_coords_history_odom = generate_lidar_history(
            scan_front_polar_coords_odom[batch_read_start:batch_read_end], history_parameters
        )
        scan_back_polar_coords_history_odom = generate_lidar_history(
            scan_back_polar_coords_odom[batch_read_start:batch_read_end], history_parameters
        )

        scan_front_polar_coords_history_base = geom_utils.transform_polar(
            scan_front_polar_coords_history_odom,
            odom_to_base[batch_read_start:batch_read_end].unsqueeze(1),
            alignment_axis="z",
        )
        del scan_front_polar_coords_history_odom
        scan_back_polar_coords_history_base = geom_utils.transform_polar(
            scan_back_polar_coords_history_odom,
            odom_to_base[batch_read_start:batch_read_end].unsqueeze(1),
            alignment_axis="z",
        )
        del scan_back_polar_coords_history_odom

        # Polar coordinates in robot frame
        # Bucketize the front scan into the virtual scan
        scan_front_virtual_history = geom_utils.aggregate_scan_history(
            scan_front_polar_coords_history_base[..., 0],
            scan_front_polar_coords_history_base[..., 1],
            angles_virtual,
            wrap=True,
            reduce="amin",
            default=1 / _eps,
        )
        del scan_front_polar_coords_history_base
        scan_back_virtual_history = geom_utils.aggregate_scan_history(
            scan_back_polar_coords_history_base[..., 0],
            scan_back_polar_coords_history_base[..., 1],
            angles_virtual,
            wrap=True,
            reduce="amin",
            default=1 / _eps,
        )
        del scan_back_polar_coords_history_base

        scan_virtual_history_batch = torch.minimum(scan_front_virtual_history, scan_back_virtual_history)
        del scan_front_virtual_history, scan_back_virtual_history

        scan_virtual_history_batch[scan_virtual_history_batch >= 1 / _eps] = torch.nan
        sensor_utils.impute_nan(scan_virtual_history_batch)
        # Impute the range_max for the virtual scan wherever it is nan
        scan_virtual_history_batch[scan_virtual_history_batch.isnan()] = lidar_virtual["range_max"]
        scan_virtual_history_batch[scan_virtual_history_batch > lidar_virtual["range_max"]] = lidar_virtual["range_max"]

        scan_virtual_history[batch_write_start:batch_write_end] = scan_virtual_history_batch[
            batch_write_start - batch_read_start : batch_write_end
        ]

    return scan_virtual_history


def subsample_history(scan_virtual_history: np.ndarray, history_parameters: dict):
    # Determine the appropriate dimension based on input tensor shape
    history_dim = 0 if scan_virtual_history.ndim == 2 else 1

    # Calculate the required length and stride
    max_index = history_parameters["length"] * history_parameters["stride"]
    if max_index > scan_virtual_history.shape[history_dim]:
        raise ValueError(
            f"stride * history_length ({max_index}) exceeds the dataset history dimension ({scan_virtual_history.shape[history_dim]})."
        )

    # Generate indices and return the subsampled tensor
    time_indices = torch.arange(history_parameters["length"]) * history_parameters["stride"]
    result = scan_virtual_history[(slice(None),) * history_dim + (time_indices,)]
    
    # Ensure the result has the correct shape
    if history_parameters["length"] == 1 and result.ndim < 2:
        result = result[np.newaxis, ...]

    return result
