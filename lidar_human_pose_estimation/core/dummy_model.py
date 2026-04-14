from numpy import isnan
import torch
from torch.utils.data import DataLoader
from typing import Tuple
from lidar_human_pose_estimation.core.model import FCN
import torch.utils
import torch.utils.data
from scipy.stats import circmean

class DummyModel(
    torch.nn.Module,
):
    def __init__(
        self,
        presence_limits: Tuple[float, float] = (0.0, 1.0),
        distance_limits: Tuple[float, float] = (0.6, 10.0),
        orientation_limits_cosine: Tuple[float, float] = (-1, 1),
        orientation_limits_sine: Tuple[float, float] = (-1, 1),
        dataloader: DataLoader = None,
        dummy_type: str = "random",
    ) -> None:
        super().__init__()

        self.dummy_type = dummy_type
        dataloader

        if self.dummy_type == "random":
            self.presence_limits = presence_limits
            self.distance_limits = distance_limits
            self.orientation_limits_cosine = orientation_limits_cosine
            self.orientation_limits_sine = orientation_limits_sine
        elif self.dummy_type == "zero":
            pass
        elif self.dummy_type == "average":
            total_presence = 0
            total_distance = 0
            total_orientation = 0
            total_rays = 0
            total_rays_presence = 0
            for batch in dataloader:
                humans_presence_sensor = batch["humans_presence_sensor"]
                humans_distance_sensor = batch["humans_distance_sensor"]
                humans_relative_bearing_sensor = batch["humans_relative_bearing_sensor"]

                fov_mask = batch["camera_fov_mask"] > 0
                presence_mask = batch["humans_presence_sensor"] > 0

                # Apply mask and calculate stats
                total_presence += humans_presence_sensor[fov_mask].sum().item()
                total_distance += humans_distance_sensor[presence_mask].sum().item()
                ori_mean = circmean(humans_relative_bearing_sensor[presence_mask].tolist(), low = -torch.pi, high = torch.pi, nan_policy='omit')
                if not isnan(ori_mean):
                    total_orientation += ori_mean
                # total_orientation += humans_relative_bearing_sensor[presence_mask].sum().item()
                
                total_rays += fov_mask.sum().item()
                total_rays_presence += presence_mask.sum().item()

            self.average_presence = total_presence / total_rays
            self.average_distance = total_distance / total_rays_presence
            self.average_orientation = total_orientation / total_rays_presence
            print(self.average_orientation)
            self.average_orientation_cosine = torch.cos(torch.tensor(self.average_orientation))
            self.average_orientation_sine = torch.sin(torch.tensor(self.average_orientation))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_tensor_size = (x.shape[0], x.shape[2])

        if self.dummy_type == "random":
            # Randomize outputs within their respective limits
            presence = (
                torch.rand(output_tensor_size, device=x.device) * (self.presence_limits[1] - self.presence_limits[0])
                + self.presence_limits[0]
            )
            distance = (
                torch.rand(output_tensor_size, device=x.device) * (self.distance_limits[1] - self.distance_limits[0])
                + self.distance_limits[0]
            )
            random_orientations = (torch.rand(output_tensor_size, device=x.device) > .5).float() * torch.pi
            cosine = torch.cos(random_orientations)
            sine = torch.sin(random_orientations)
        elif self.dummy_type == "zero":
            presence = torch.zeros(output_tensor_size, device=x.device)
            distance = torch.zeros(output_tensor_size, device=x.device)
            cosine = torch.zeros(output_tensor_size, device=x.device)
            sine = torch.zeros(output_tensor_size, device=x.device)
        elif self.dummy_type == "average":
            presence = torch.ones(output_tensor_size, device=x.device) * self.average_presence
            distance = torch.ones(output_tensor_size, device=x.device) * self.average_distance
            cosine = torch.ones(output_tensor_size, device=x.device) * self.average_orientation_cosine
            sine = torch.ones(output_tensor_size, device=x.device) * self.average_orientation_sine

        return dict(
            presence=presence,
            distance=distance,
            cosine=cosine,
            sine=sine,
        )


if __name__ == "__main__":
    fcn_configs = {
        "input_channels": 10,
        "use_skip_connection": True,
        "layer_configs": [
            {"out_channels": 16, "kernel_size": 3, "dilation": 1, "use_group_norm": False},
            {"out_channels": 16, "kernel_size": 3, "dilation": 2, "use_group_norm": False},
            {"out_channels": 16, "kernel_size": 5, "dilation": 2, "use_group_norm": True},
            {"out_channels": 16, "kernel_size": 5, "dilation": 2, "use_group_norm": False},
            {"out_channels": 16, "kernel_size": 5, "dilation": 2, "use_group_norm": False},
        ],
    }

    model = FCN(
        input_channels=fcn_configs["input_channels"],
        use_skip_connection=fcn_configs["use_skip_connection"],
        layer_configs=fcn_configs["layer_configs"],
    )
