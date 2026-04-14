import torch
from typing import List, Union
from lidar_human_pose_estimation.core.config import _d_max, Batch


class FCN(torch.nn.Module):
    def __init__(
        self,
        layer_configs: List[dict],  # List of dictionaries specifying layer configuration
        input_channels: int = 25,
        loss_activation: dict = None,
        use_skip_connection: bool = False,
    ) -> None:
        super().__init__()

        layers = []
        in_channels = input_channels
        self.use_skip_connection = use_skip_connection

        self.presence_active = loss_activation["presence"]
        self.distance_active = loss_activation["distance"]
        self.orientation_active = loss_activation["orientation"]
        self.verse_active = loss_activation["verse"]
        output_channels = self.get_output_channels()

        # Create network
        for layer_config in layer_configs:
            layers.append(
                torch.nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=layer_config["out_channels"],
                    kernel_size=layer_config["kernel_size"],
                    dilation=layer_config["dilation"],
                    padding=0,
                )
            )

            if layer_config["use_group_norm"]:
                layers.append(torch.nn.GroupNorm(num_groups=1, num_channels=layer_config["out_channels"]))
            layers.append(torch.nn.GELU())
            in_channels = layer_config["out_channels"]

        self.layers = torch.nn.Sequential(*layers)

        if self.use_skip_connection:
            in_channels += 1

        # Final layer to map to output_channels
        self.last_layer = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=output_channels, kernel_size=1, dilation=1, padding=0
        )

        # Compute receptive field
        input_size = 360
        with torch.no_grad():
            if self.use_skip_connection:
                result = self.layers(torch.randn(1, input_channels, input_size))
                result = torch.cat([result, torch.randn(1, 1, result.shape[-1])], dim=1)
                output_size = self.last_layer(result).size(-1)
            else:
                output_size = self.last_layer(self.layers(torch.randn(1, input_channels, input_size))).size(-1)

        self.receptive_field = input_size - output_size + 1

        print(f"Receptive field: {self.receptive_field}")

    def get_output_channels(self) -> int:
        output_layers = 0
        self.nn_channel_to_output = {}
        if self.presence_active:
            self.nn_channel_to_output["presence"] = output_layers
            output_layers += 1
        if self.distance_active:
            self.nn_channel_to_output["distance"] = output_layers
            output_layers += 1
        if self.orientation_active:
            self.nn_channel_to_output["cosine"] = output_layers
            output_layers += 1
            self.nn_channel_to_output["sine"] = output_layers
            output_layers += 1
        if self.verse_active:
            self.nn_channel_to_output["verse"] = output_layers
            output_layers += 1
        return output_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padding = self.receptive_field // 2
        x_padded = torch.cat([x[..., -padding:], x, x[..., :padding]], dim=-1)

        result = self.layers(x_padded)
        if self.use_skip_connection:
            result = torch.cat([result, x[:, 0, :].unsqueeze(1)], dim=1)
        result = self.last_layer(result)

        result_dict = {}
        if self.presence_active:
            result_dict["presence"] = torch.sigmoid(result[..., self.nn_channel_to_output["presence"], :])
        if self.distance_active:
            result_dict["distance"] = torch.sigmoid(result[..., self.nn_channel_to_output["distance"], :]) * _d_max
        if self.orientation_active:
            result_dict["cosine"] = result[..., self.nn_channel_to_output["cosine"], :]
            result_dict["sine"] = result[..., self.nn_channel_to_output["sine"], :]
        if self.verse_active:
            result_dict["verse"] = result[..., self.nn_channel_to_output["verse"], :]

        return result_dict

        # Comment the return stmnt above for torchscan.summary
        return result

class AttnFCN(torch.nn.Module):
    def __init__(
        self,
        layer_configs: List[dict],  # List of dictionaries specifying layer configuration
        input_channels: int = 25,
        loss_activation: dict = None,
        use_skip_connection: bool = False,
    ) -> None:
        super().__init__()

        history_size = input_channels
        self.history_size = history_size
        self.presence_active = loss_activation["presence"]
        self.distance_active = loss_activation["distance"]
        self.orientation_active = loss_activation["orientation"]
        output_channels = self.get_output_channels()


        self.layers = [
            torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.LazyBatchNorm2d(),
            )
        ]
        attention_size = 32

        kernel_sizes = [3, 5, 5, 5, 5]
        for i, k in enumerate(kernel_sizes):
            self.layers.append(torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=2 ** (i + 1), out_channels=2 ** (i + 2), kernel_size=k, padding = k // 2),
                torch.nn.GELU(),
                torch.nn.LazyBatchNorm2d()
            ))
        self.layers.append(
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=2 ** (len(kernel_sizes) + 1), out_channels=4, kernel_size=k, padding = k // 2),
                torch.nn.LazyBatchNorm2d()
            )
        )

        self.layers = torch.nn.Sequential(*self.layers)
        self.final_layers = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=4 * self.history_size, out_channels=4 * self.history_size // 2, kernel_size=5, padding = 2),
            torch.nn.GELU(),
            torch.nn.LazyBatchNorm1d(),
            torch.nn.Conv1d(in_channels=4 * self.history_size // 2, out_channels=4, kernel_size=5, padding = 2),
        )

    
    def forward(self, x : torch.Tensor):
        x = x[:, None, ...]
        result = self.layers(x)

        attn = result[:, :1, ...].sum(1, keepdims = True).sigmoid()
        attended_features = result[:, 1:, ...].mul(attn).flatten(1, 2)
        attn = attn.flatten(1, 2)
        result = torch.cat([attn, attended_features], dim = 1)
        result = self.final_layers(result)

        result_dict = {}

        if self.presence_active:
            result_dict["presence"] = torch.sigmoid(result[..., self.nn_channel_to_output["presence"], :])
        if self.distance_active:
            result_dict["distance"] = torch.sigmoid(result[..., self.nn_channel_to_output["distance"], :]) * _d_max
        if self.orientation_active:
            result_dict["cosine"] = result[..., self.nn_channel_to_output["cosine"], :]
            result_dict["sine"] = result[..., self.nn_channel_to_output["sine"], :]

        return result_dict

    def get_output_channels(self) -> int:
        output_layers = 0
        self.nn_channel_to_output = {}
        if self.presence_active:
            self.nn_channel_to_output["presence"] = output_layers
            output_layers += 1
        if self.distance_active:
            self.nn_channel_to_output["distance"] = output_layers
            output_layers += 1
        if self.orientation_active:
            self.nn_channel_to_output["cosine"] = output_layers
            output_layers += 1
            self.nn_channel_to_output["sine"] = output_layers
            output_layers += 1
        return output_layers

class LHPELossFunction:
    def __init__(
        self,
        presence: bool = True,
        distance: bool = True,
        orientation: bool = True,
        bidirection: bool = False,
        verse: bool = False,
    ) -> None:
        super().__init__()

        self.presence = presence
        self.distance = distance
        self.orientation = orientation
        self.bidirection = bidirection
        self.verse = verse

    def loss_function(self, pred: Batch, gt: Batch) -> Batch:
        # Kinect FOV
        fov_mask = gt["camera_fov_mask"] > 0
        # Presence mask
        presence_mask = fov_mask & (gt["humans_presence_sensor"] > 0)

        loss_dict = {}

        if self.presence:
            # Loss for presence using FOV mask
            loss_presence_raw = torch.nn.functional.mse_loss(
                pred["presence"][fov_mask], gt["humans_presence_sensor"][fov_mask], reduction="none"
            )
            loss_presence_raw[presence_mask[fov_mask]] *= 10
            loss_presence = loss_presence_raw.mean()
            loss_dict["loss_presence"] = loss_presence

        if not presence_mask.any():
            return loss_dict

        if self.distance:
            # Loss for distance using GT presence mask
            loss_distance = torch.nn.functional.mse_loss(
                pred["distance"][presence_mask], gt["humans_distance_sensor"][presence_mask]
            )
            loss_dict["loss_distance"] = loss_distance

        if self.orientation:
            gt_sine = torch.sin(gt["humans_relative_bearing_sensor"][presence_mask])
            gt_cosine = torch.cos(gt["humans_relative_bearing_sensor"][presence_mask])

            if self.bidirection:
                # Calculate opposite loss and take the minimum
                # loss_orientation_opposite = 0.5 * (
                #     torch.nn.functional.mse_loss(pred["sine"][presence_mask], -gt_sine, reduction="none")
                #     + torch.nn.functional.mse_loss(pred["cosine"][presence_mask], -gt_cosine, reduction="none")
                # )
                # loss_orientation = torch.minimum(loss_orientation, loss_orientation_opposite)

                # Angle between -pi/2 and pi/2
                humans_direction_angle = torch.arcsin(torch.sin(gt["humans_relative_bearing_sensor"][presence_mask]))
                # Loss for orientation using GT presence mask
                gt_sine_direction = torch.sin(2.0 * humans_direction_angle)
                gt_cosine_direction = torch.cos(2.0 * humans_direction_angle)
                loss_dict["loss_orientation"] = 0.5 * (
                    torch.nn.functional.mse_loss(pred["sine"][presence_mask], gt_sine_direction, reduction="mean")
                    + torch.nn.functional.mse_loss(pred["cosine"][presence_mask], gt_cosine_direction, reduction="mean")
                )
                if self.verse:
                    loss_dict["loss_verse"] = torch.nn.functional.mse_loss(
                        pred["verse"][presence_mask], gt_cosine, reduction="mean"
                    )
            else:
                # Loss for orientation using GT presence mask
                loss_dict["loss_orientation"] = 0.5 * (
                    torch.nn.functional.mse_loss(pred["sine"][presence_mask], gt_sine, reduction="mean")
                    + torch.nn.functional.mse_loss(pred["cosine"][presence_mask], gt_cosine, reduction="mean")
                )
        return loss_dict


if __name__ == "__main__":
    import argparse
    import torchscan

    parser = argparse.ArgumentParser()
    parser.add_argument("input_shape", nargs="+", type=int)
    args = parser.parse_args()

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

    torchscan.summary(model, input_shape=args.input_shape)
