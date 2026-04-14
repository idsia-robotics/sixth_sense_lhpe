import torch
from lidar_human_pose_estimation.utils import geom_utils
from lidar_human_pose_estimation.core.config import _eps, Sensor


def get_sensor_angles(sensor: Sensor, device: str = "cpu", eps: float = _eps) -> torch.Tensor:
    return torch.arange(
        sensor["angle_min"], sensor["angle_max"] + eps, sensor["angle_increment"], dtype=torch.float64, device=device
    )


def transform_scan_to_cartesian(
    scan: torch.Tensor, angle: torch.Tensor, pose: torch.Tensor, alignment_axis: str
) -> torch.Tensor:
    polar_coords = torch.stack([scan, angle.tile(scan.shape[0], 1)], dim=-1)
    polar_coords = geom_utils.transform_polar(polar_coords, pose, alignment_axis)
    cartesian_coords = geom_utils.polar_to_cartesian(polar_coords)
    return cartesian_coords


def cartesian_range(sensor: Sensor, pose: torch.Tensor, alignment_axis: str, device: str = "cpu") -> torch.Tensor:
    angle = get_sensor_angles(sensor)
    scan_min = torch.tensor([[sensor["range_min"]]], device=device).tile(pose.size(0), angle.size(0))
    cartesian_min = transform_scan_to_cartesian(scan_min, angle, pose, alignment_axis)
    scan_max = torch.tensor([[sensor["range_max"]]], device=device).tile(pose.size(0), angle.size(0))
    cartesian_max = transform_scan_to_cartesian(scan_max, angle.flip(0), pose, alignment_axis)
    return torch.cat([cartesian_min, cartesian_max, cartesian_min[:, :1]], dim=1)


def impute_nan(x: torch.Tensor) -> None:
    folds = torch.cat([x[..., -1:], x, x[..., :1]], dim=-1).unfold(dimension=-1, size=3, step=1)
    mask = (~folds[..., 0].isnan()) & folds[..., 1].isnan() & (~folds[..., 2].isnan())
    x[mask] = folds[mask][..., [0, 2]].mean(dim=-1)
