import pandas as pd
from matplotlib import colormaps
import base64
import io
import h5py
from numbers import Number
import numpy as np
from PIL import Image
import pathlib
from lidar_human_pose_estimation.core.metrics import circular_pearson
import plotly.graph_objects as go
import matplotlib.colors as mcolors
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from lidar_human_pose_estimation.utils import (
    sensor_utils,
    gt_utils,
    geom_utils,
    shell_utils,
    temporal_registration_utils,
)
from sklearn.metrics import PrecisionRecallDisplay


def plotly_fig_to_pil_image(fig: go.Figure) -> Image.Image:
    """Converts a plotly figure in a PIL image of RGB pixels."""
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    return Image.open(buf)


def pil_image_to_array(image: Image.Image) -> np.ndarray:
    """Converts a PIL image of RGB pixels in a numpy matrix of RGB pixels and shape (h, w, c)."""
    return np.asarray(image, dtype=np.uint8)[..., :3]


def pil_image_to_base64(image: Image.Image) -> str:
    """Converts a PIL image of RGB pixels in the base64 string representation."""
    buf = io.BytesIO()
    image.save(buf, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def plotly_fig_to_array(fig: go.Figure) -> np.ndarray:
    """Converts a plotly figure in a numpy matrix of RGB pixels and shape (h, w, c)."""
    return pil_image_to_array(plotly_fig_to_pil_image(fig))


def rotate_and_save_images(head_yaws: torch.Tensor, image_path: str, folder_path: str):
    if folder_path.exists():
        shell_utils.cmd("rm", "-r", str(folder_path))
    folder_path.mkdir()
    # Load the image from the given path
    image = Image.open(image_path)

    # Convert folder_path to Path object
    folder_path = pathlib.Path(folder_path)

    # Get min and max angles
    min_angle = torch.min(head_yaws).item()
    max_angle = torch.max(head_yaws).item()

    # Iterate through angles and save images
    for angle in range(min_angle, max_angle + 1):
        # Rotate image using PyTorch
        rotated_image = TF.rotate(image, angle)

        # Save image
        output_path = folder_path / f"image_{angle}.png"
        rotated_image.save(output_path)


##############


def make_humans_plot(name: str, color: str, alpha: float = 1.0, marker_size: int = 10) -> go.Scatter:
    # rgba = mcolors.to_rgba(color)
    # rgba = f"rgba({rgba[0]},{rgba[1]},{rgba[2]},{alpha})"
    return go.Scatter(
        x=[0],
        y=[0],
        mode="lines+markers",
        name=name,
        marker=dict(color=color, size=marker_size, symbol="arrow-up", angleref="previous"),
        line=dict(color=color, width=int(marker_size / 5)),
        legendgroup="Humans",
        legendgrouptitle={"text": "Humans"},
    )


def make_model_plot(name: str, color: str, alpha: float = 0.5, marker_size: int = 2) -> go.Scatter:
    rgba = mcolors.to_rgba(color)

    color = f"rgba({rgba[0]},{rgba[1]},{rgba[2]},{alpha})"
    return go.Scatter(
        x=[0],
        y=[0],
        mode="lines",
        name=name,
        marker=dict(color=color, size=marker_size),
        line=dict(color=color, width=marker_size),
        legendgroup="Model Output",
        legendgrouptitle={"text": "Model Output"},
    )


def make_model_range_plot(name: str, color: str, alpha: float = 1.0, marker_size: int = 2) -> go.Scatter:
    return go.Scatter(
        x=[0],
        y=[0],
        mode="lines",
        name=name,
        marker=dict(color=color, size=marker_size, opacity=alpha),
        line=dict(dash="dot", width=marker_size),
        legendgroup="Model Output",
        legendgrouptitle={"text": "Model Output"},
    )


def make_sensor_plot(name: str, color: str, marker_size: int = 6) -> go.Scatter:
    return go.Scatter(
        x=[0],
        y=[0],
        mode="markers",
        name=name,
        marker=dict(color=color, opacity=0.5, size=marker_size),
        legendgroup="Sensor",
        legendgrouptitle={"text": "Sensor"},
    )


def make_sensor_range_plot(name: str, color: str, alpha: float = 1.0) -> go.Scatter:
    return go.Scatter(
        x=[0],
        y=[0],
        mode="lines",
        name=name,
        fill="toself",
        line_color=color,
        line_width=0,
        opacity=alpha,
        legendgroup="Sensor",
        legendgrouptitle={"text": "Sensor"},
    )


def make_image_plot(name: str, image: Image.Image, x: Number, y: Number, w: Number, h: Number) -> go.Image:
    return go.Image(
        name=name,
        x0=x - w / 2,
        y0=y - h / 2,
        dx=w / image.size[0],
        dy=h / image.size[1],
        hoverinfo="name",
        source=pil_image_to_base64(image),
    )


def substitute_limits_nans_with_value(tensor, value):
    # Detect NaN values
    is_nan = torch.isnan(tensor)

    # Shift left and right to check neighbors
    left_is_nan = torch.cat([is_nan[:, 1:], is_nan[:, 0].unsqueeze(-1)], dim=1)
    right_is_nan = torch.cat([is_nan[:, -1].unsqueeze(-1), is_nan[:, :-1]], dim=1)

    # Condition: Element is NaN and at least one neighbor is not NaN
    condition = is_nan & ~(left_is_nan & right_is_nan)

    # Replace NaN values with nms_threshold where the condition is True
    tensor[condition] = value

    return tensor


def read_h5_file(h5_file: str, device: str = "cpu") -> dict:
    sensors_info = {}
    with h5py.File(h5_file, mode="r") as h5f:
        sensors_info["front"] = {
            "angle_min": h5f.attrs["scan_raw_angle_min"],
            "angle_max": h5f.attrs["scan_raw_angle_max"],
            "range_min": h5f.attrs["scan_raw_range_min"],
            "range_max": h5f.attrs["scan_raw_range_max"],
            "angle_increment": h5f.attrs["scan_raw_angle_increment"],
        }
        sensors_info["back"] = {
            "angle_min": h5f.attrs["scan_raw_back_angle_min"],
            "angle_max": h5f.attrs["scan_raw_back_angle_max"],
            "range_min": h5f.attrs["scan_raw_back_range_min"],
            "range_max": h5f.attrs["scan_raw_back_range_max"],
            "angle_increment": h5f.attrs["scan_raw_back_angle_increment"],
        }
        sensors_info["virtual"] = {
            "angle_min": h5f.attrs["scan_virtual_angle_min"],
            "angle_max": h5f.attrs["scan_virtual_angle_max"],
            "range_min": h5f.attrs["scan_virtual_range_min"],
            "range_max": h5f.attrs["scan_virtual_range_max"],
            "angle_increment": h5f.attrs["scan_virtual_angle_increment"],
        }
        sensors_info["kinect"] = {
            "angle_min": h5f.attrs["azure_kinect_angle_min"],
            "angle_max": h5f.attrs["azure_kinect_angle_max"],
            "range_min": h5f.attrs["azure_kinect_range_min"],
            "range_max": h5f.attrs["azure_kinect_range_max"],
            "angle_increment": h5f.attrs["azure_kinect_angle_increment"],
        }

        # Utils for sensors
        sensors_angles = {}
        sensors_angles["front"] = sensor_utils.get_sensor_angles(sensors_info["front"])
        sensors_angles["back"] = sensor_utils.get_sensor_angles(sensors_info["back"])
        sensors_angles["virtual"] = sensor_utils.get_sensor_angles(sensors_info["virtual"])
        sensors_angles["kinect"] = sensor_utils.get_sensor_angles(sensors_info["kinect"])

        # Sensors data
        sensors_scans = {}
        sensors_scans["front"] = torch.tensor(h5f["scan_raw"][:], dtype=torch.float64, device=device)
        sensors_scans["back"] = torch.tensor(h5f["scan_raw_back"][:], dtype=torch.float64, device=device)
        sensors_scans["virtual_history"] = torch.tensor(
            h5f["scan_virtual_history"][:], dtype=torch.float64, device=device
        )

        for sensor_scan in sensors_scans.keys():
            sensors_scans[sensor_scan] = sensors_scans[sensor_scan]

        sensors_scans["virtual"] = sensors_scans["virtual_history"][:, 0, :]

        # Sensors poses
        pose_lidar_front = torch.tensor(h5f["tf_base_laser_link_wrt_base_link"][:], dtype=torch.float64, device=device)
        pose_lidar_back = torch.tensor(
            h5f["tf_base_laser_back_link_wrt_base_link"][:], dtype=torch.float64, device=device
        )
        pose_kinect = torch.tensor(
            h5f["tf_azure_kinect_depth_camera_link_wrt_base_link"][:], dtype=torch.float64, device=device
        )
        sensors_poses = {}
        sensors_poses["front"] = geom_utils.pose_to_matrix(pose_lidar_front).unsqueeze(1)
        sensors_poses["back"] = geom_utils.pose_to_matrix(pose_lidar_back).unsqueeze(1)
        sensors_poses["virtual"] = torch.eye(4, dtype=torch.float64, device=device)[None, None].tile(
            pose_lidar_front.size(0), 1, 1, 1
        )
        sensors_poses["virtual_history"] = torch.eye(4, dtype=torch.float64, device=device)[None, None].tile(
            sensors_poses["front"].shape[0], sensors_scans["virtual_history"].shape[1], 1, 1
        )
        sensors_poses["kinect"] = geom_utils.pose_to_matrix(pose_kinect).unsqueeze(1)

        for sensor in sensors_poses.keys():
            sensors_poses[sensor] = sensors_poses[sensor]

        # Ground truth
        ground_truth = {}
        ground_truth["fov_mask"] = torch.tensor(h5f["camera_fov_mask"][:], dtype=torch.float64, device=device)
        ground_truth["presence"] = gt_utils.circular_erosion(
            torch.tensor(h5f["humans_presence_sensor"][:], dtype=torch.float64, device=device)
        )

        ground_truth["distance"] = torch.tensor(h5f["humans_distance_sensor"][:], dtype=torch.float64, device=device)
        ground_truth["orientation_rel"] = torch.tensor(h5f["humans_relative_bearing_sensor"][:], dtype=torch.float64, device=device)
        ground_truth["orientation_abs"] = gt_utils.absolute_bearing(
            ground_truth["orientation_rel"], sensors_angles["virtual"]
        )

    return sensors_info, sensors_angles, sensors_scans, sensors_poses, ground_truth


def read_h5_file_optitrack(h5_file: str, device: str = "cpu") -> dict:
    with h5py.File(h5_file, mode="r") as h5f:
        sensors_info = {}
        sensors_info_virtual = {
            "angle_min": h5f.attrs["scan_virtual_angle_min"],
            "angle_max": h5f.attrs["scan_virtual_angle_max"],
            "range_min": h5f.attrs["scan_virtual_range_min"],
            "range_max": h5f.attrs["scan_virtual_range_max"],
            "angle_increment": h5f.attrs["scan_virtual_angle_increment"],
        }

        sensors_angles_virtual = sensor_utils.get_sensor_angles(sensors_info_virtual)

        if "humans_presence_optitrack" not in h5f:
            return None

        # Ground truth
        ground_truth_optitrack = {}
        ground_truth_optitrack["presence"] = gt_utils.circular_erosion(
            torch.tensor(h5f["humans_presence_optitrack"][:], dtype=torch.float64, device=device)
        )

        ground_truth_optitrack["distance"] = torch.tensor(
            h5f["humans_distance_optitrack"][:], dtype=torch.float64, device=device
        )
        ground_truth_optitrack["orientation_rel"] = torch.tensor(
            h5f["humans_relative_bearing_optitrack"][:], dtype=torch.float64, device=device
        )
        ground_truth_optitrack["orientation_abs"] = gt_utils.absolute_bearing(
            ground_truth_optitrack["orientation_rel"], sensors_angles_virtual
        )

        return ground_truth_optitrack


def human_data_to_arrows(humans_presence_sensor, humans_distance_sensor, humans_orientation_abs, sensor_angles, sensor_poses):
    mask_humans_presence = humans_presence_sensor > 0
    humans_coordinates = sensor_utils.transform_scan_to_cartesian(
        humans_presence_sensor * humans_distance_sensor, sensor_angles, sensor_poses, alignment_axis="z"
    )

    humans_arrows_origins = torch.nested.to_padded_tensor(
        torch.nested.nested_tensor(
            [humans_coordinates[i, mask_humans_presence[i]] for i in range(humans_coordinates.size(0))],
            dtype=torch.float64,
        ),
        padding=torch.nan,
    )
    humans_orientations = torch.nested.to_padded_tensor(
        torch.nested.nested_tensor(
            [humans_orientation_abs[i, mask_humans_presence[i]] for i in range(humans_coordinates.size(0))],
            dtype=torch.float64,
        ),
        padding=torch.nan,
    )
    humans_arrows_direction = 0.8 * torch.stack(
        [torch.cos(humans_orientations), torch.sin(humans_orientations)], dim=-1
    )
    humans_arrows = torch.stack(
        [
            humans_arrows_origins,
            humans_arrows_origins + humans_arrows_direction,
            torch.full_like(humans_arrows_origins, torch.nan),
        ],
        dim=-1,
    )

    return humans_arrows


def arrows_to_human_detections(humans_arrows):
    """
    Extracts (x, y, theta) tuples of valid detections from humans_arrows.

    Args:
        humans_arrows (torch.Tensor): Tensor containing human arrow data with shape (num_humans, 3, 2)

    Returns:
        list of tuples: A list containing (x, y, theta) tuples for each valid detection.
    """
    valid_detections = []

    for arrow in humans_arrows:
        if not torch.isnan(arrow).any():  # Ensure valid detection (no NaNs)
            origin = arrow[:, 0]  # (x, y) position
            direction = arrow[:, 1] - origin  # Direction vector
            theta = torch.atan2(direction[1], direction[0])  # Compute orientation
            valid_detections.append((origin[0].item(), origin[1].item(), theta.item()))

    return valid_detections


def human_detections_to_arrows(human_detections, shape):
    """
    Converts a list of (x, y, theta) human detections into arrow representation.

    Args:
        human_detections (list of tuples): List containing (x, y, theta) tuples.
        shape (tuple): Desired shape for the output tensor.

    Returns:
        torch.Tensor: Tensor containing human arrow data with the specified shape.
    """
    # Create the output tensor filled with NaNs
    humans_arrows_tensor = torch.full(shape, torch.nan, dtype=torch.float64)

    # Ensure all elements are tuples of length 3
    filtered_detections = []
    for item in human_detections:
        if isinstance(item, (list, tuple)) and len(item) == 3:
            filtered_detections.append(tuple(item))
        else:
            print(f"Warning: Skipping invalid detection {item}")

    if not filtered_detections:  # Handle empty input case
        return humans_arrows_tensor

    arrow_length = 0.8  # Same scaling factor used in human_data_to_arrows

    for i, (x, y, theta) in enumerate(filtered_detections):
        if i >= shape[0]:  # Prevent exceeding the preallocated shape
            break
        origin = torch.tensor([x, y], dtype=torch.float64)
        direction = arrow_length * torch.tensor(
            [torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))], dtype=torch.float64
        )
        endpoint = origin + direction
        humans_arrows_tensor[i, :2, 0] = origin
        humans_arrows_tensor[i, :2, 1] = endpoint

    return humans_arrows_tensor


def model_prediction_scatter_plot(ground_truth: dict, pred: dict, output: pathlib.Path):
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    presence_mask = ground_truth["presence"] == 1
    scatter_plot_ori_gt = ground_truth["orientation_rel"][presence_mask].numpy()
    scatter_plot_ori_pred = torch.atan2(pred["sine"], pred["cosine"])[presence_mask].numpy()
    scatter_plot_distance_gt = ground_truth["distance"][presence_mask].numpy()
    scatter_plot_distance_pred = pred["distance"][presence_mask].numpy()

    scatter_plot_sine_gt = np.sin(scatter_plot_ori_gt)
    scatter_plot_cosine_gt = np.cos(scatter_plot_ori_gt)
    scatter_plot_sine_pred = pred["sine"][presence_mask].numpy()
    scatter_plot_cosine_pred = pred["cosine"][presence_mask].numpy()

    orientation_error = scatter_plot_ori_pred - scatter_plot_ori_gt
    distance_error = scatter_plot_distance_pred - scatter_plot_distance_gt
    orientation_error = (orientation_error + np.pi) % (2 * np.pi) - np.pi

    # Create the scatter plots
    fig, axes = plt.subplots(5, 2, figsize=(12, 24), dpi=300)

    # Plot for distance
    axes[0, 0].scatter(scatter_plot_distance_gt, scatter_plot_distance_pred, color="blue", alpha=0.2, label="Distance")
    axes[0, 0].plot(
        [scatter_plot_distance_gt.min(), scatter_plot_distance_gt.max()],
        [scatter_plot_distance_gt.min(), scatter_plot_distance_gt.max()],
        color="red",
        linestyle="--",
        label="Diagonal",
    )  # Diagonal line
    dist_pearson = np.corrcoef(scatter_plot_distance_gt, scatter_plot_distance_pred)[0, 1]
    axes[0, 0].set_title(f"Predicted vs Ground Truth (Distance)\nPearson:{round(dist_pearson.item(), 2)}")
    axes[0, 0].set_xlabel("Ground Truth Distance")
    axes[0, 0].set_ylabel("Predicted Distance")
    axes[0, 0].axis("equal")
    axes[0, 0].legend()

    # Plot for orientation
    axes[0, 1].scatter(
        scatter_plot_ori_gt, scatter_plot_ori_pred, color="green", alpha=0.15, label="Orientation", edgecolor="none"
    )
    axes[0, 1].plot(
        [scatter_plot_ori_gt.min(), scatter_plot_ori_gt.max()],
        [scatter_plot_ori_gt.min(), scatter_plot_ori_gt.max()],
        color="red",
        linestyle="--",
        label="Diagonal",
    )  # Diagonal line
    ori_pearson = circular_pearson(scatter_plot_ori_gt, scatter_plot_ori_pred)

    axes[0, 1].set_title(f"Predicted vs Ground Truth (Orientation)\nCirc. Pearson:{round(ori_pearson.item(), 2)}")
    axes[0, 1].set_xlabel("Ground Truth Orientation")
    axes[0, 1].set_ylabel("Predicted Orientation")
    axes[0, 1].axis("equal")
    axes[0, 1].legend()

    # Plot error for distance
    axes[1, 0].scatter(scatter_plot_distance_gt, distance_error, color="purple", alpha=0.2, label="Distance Error")
    axes[1, 0].axhline(0, color="red", linestyle="--", label="Zero Error Line")
    axes[1, 0].set_title("Prediction Error vs Ground Truth (Distance)")
    axes[1, 0].set_xlabel("Ground Truth Distance")
    axes[1, 0].set_ylabel("Error (Predicted - Ground Truth)")
    axes[1, 0].legend()

    # Plot error for orientation
    axes[1, 1].scatter(
        scatter_plot_ori_gt, orientation_error, color="orange", alpha=0.15, label="Orientation Error", edgecolor="none"
    )
    axes[1, 1].axhline(0, color="red", linestyle="--", label="Zero Error Line")
    axes[1, 1].set_title("Prediction Error vs Ground Truth (Orientation)")
    axes[1, 1].set_xlabel("Ground Truth Orientation")
    axes[1, 1].set_ylabel("Error (Predicted - Ground Truth)")
    axes[1, 1].legend()

    # Combined plot for predicted sine and cosine

    scatter = axes[2, 0].scatter(
        scatter_plot_cosine_pred,
        scatter_plot_sine_pred,
        c=np.abs(orientation_error),
        alpha=0.3,
        label="Predicted",
        edgecolor="none",
        cmap="cool",
        vmin=0,
        vmax=np.pi,
    )

    axes[2, 0].set_title("Predicted Sine vs Cosine")
    axes[2, 0].set_ylabel("Predicted Sine")
    axes[2, 0].set_xlabel("Predicted Cosine")
    axes[2, 0].set_xlim(-1.1, 1.1)
    axes[2, 0].set_ylim(-1.1, 1.1)
    fig.colorbar(scatter, ax=axes[2, 0], label="error")

    # Combined plot for ground truth sine and cosine
    axes[2, 1].scatter(
        scatter_plot_cosine_gt, scatter_plot_sine_gt, color="magenta", alpha=0.3, label="Ground Truth", edgecolor="none"
    )
    axes[2, 1].set_title("Ground Truth Sine vs Cosine")
    axes[2, 1].set_ylabel("Ground Truth Sine")
    axes[2, 1].set_xlabel("Ground Truth Cosine")
    axes[2, 1].set_xlim(-1.1, 1.1)
    axes[2, 1].set_ylim(-1.1, 1.1)
    axes[2, 1].legend()

    orientation_error_bins = np.arange(0, np.pi, 0.1)
    axes[3, 0].hist(np.abs(orientation_error), bins=orientation_error_bins)
    axes[3, 0].set_title("Orientation error distribution")
    axes[3, 0].set_ylabel("Frequency")
    axes[3, 0].set_xlabel("Error")

    range = [[-1, 1], [-1, 1]]
    (
        heatmap,
        xedges,
        yedges,
    ) = np.histogram2d(
        scatter_plot_sine_pred, scatter_plot_cosine_pred, bins=[30, 30], range=range, weights=np.abs(orientation_error)
    )

    (
        counts,
        _,
        __,
    ) = np.histogram2d(
        scatter_plot_sine_pred,
        scatter_plot_cosine_pred,
        bins=[30, 30],
        range=range,
    )

    heatmap /= counts
    heatmap = np.ma.masked_invalid(heatmap)
    heatmap = np.flip(heatmap, axis=0)
    img = axes[3, 1].imshow(heatmap, cmap="cool", extent=[-1, 1, -1, 1])
    axes[3, 1].set_title("Orientation Prediction Error Heatmap")
    axes[3, 1].set_ylabel("Predicted Sine")
    axes[3, 1].set_xlabel("Predicted Cosine")
    fig.colorbar(img, ax=axes[3, 1])

    PrecisionRecallDisplay.from_predictions(
        ground_truth["presence"].flatten().long().numpy(),
        pred["presence"].flatten().numpy(),
        ax=axes[4, 0],
        plot_chance_level=True,
    )
    axes[4, 0].set_title("Presence Precision-Recall curve")

    # Adjust layout and save
    plt.tight_layout()
    fig_name = str(output.parent / output.stem)
    plt.savefig(fig_name + ".png")
    # plt.savefig(fig_name + ".svg")
    # plt.savefig(fig_name + ".pdf")
    # plt.show()
