import torch
import functools
from typing import Callable, List, Optional
from lidar_human_pose_estimation.core.config import ArrayLike, Batch, Transform


def to_tensor(x: ArrayLike, device: str = "cpu", dtype: torch.FloatType = torch.float32) -> torch.Tensor:
    return torch.tensor(x, device=device, dtype=dtype).contiguous()


def multiplicative_noise(x: torch.Tensor, std: float = 0.05) -> torch.Tensor:
    n = 1 + torch.randn_like(x) * std
    return x * n.clamp(0, 1)

def additive_noise(x : torch.Tensor, std: float = .03) -> torch.Tensor:
    n = torch.randn_like(x) * std
    return x.add(n)

def flip(data: Batch, p: float = 0.5, dim: int = -1, keys: Optional[List[str]] = None) -> Batch:
    if keys is None:
        keys = data.keys()

    if torch.rand(1) < p:
        for k in keys:
            data[k] = data[k].flip(dim)
            if k == 'humans_relative_bearing_sensor':
                data[k] = data[k].neg()
    return data


def batch_apply(data: Batch, fn: Callable[[ArrayLike], torch.Tensor], keys: Optional[List[str]] = None) -> Batch:
    if keys is None:
        keys = data.keys()

    return {k: fn(v) if k in keys else v for k, v in data.items()}


##############


def get_transform(augment: bool) -> Transform:
    keys_all = None
    keys_input = ["scan_virtual_history"]
    keys_gt = ["camera_fov_mask", "humans_presence_sensor", "humans_relative_bearing_sensor", "humans_distance_sensor"]

    if not augment:
        return functools.partial(batch_apply, fn=to_tensor, keys=keys_all)

    return functools.reduce(
        lambda fa, fb: lambda x: fb(fa(x)),
        [
            functools.partial(batch_apply, fn=to_tensor, keys=keys_all),
            functools.partial(batch_apply, fn=additive_noise, keys=keys_input),
            functools.partial(flip, keys=keys_input + keys_gt),
        ],
    )
