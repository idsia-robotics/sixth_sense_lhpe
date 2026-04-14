import tqdm
import torch
import plotly.graph_objects as go
from lidar_human_pose_estimation.utils import shell_utils
from lidar_human_pose_estimation.utils import sensor_utils
from lidar_human_pose_estimation.utils import vis_utils
from lidar_human_pose_estimation.core.config import parse_args

args = parse_args("input", "output")
input_path = args.input
output_path = args.output
device = "cpu"

tmp_folder = output_path.parent / "tmp"
# If tmp_folder exists delete it
if tmp_folder.exists():
    shell_utils.cmd("rm", "-r", str(tmp_folder))
tmp_folder.mkdir()

sensors_info, sensors_angles, sensors_scans, sensors_poses, ground_truth = vis_utils.read_h5_file(input_path, device)

# Prepare scan readings for plotting
coords_front = sensor_utils.transform_scan_to_cartesian(
    sensors_scans["front"], sensors_angles["front"], sensors_poses["front"], alignment_axis="z"
)
coords_back = sensor_utils.transform_scan_to_cartesian(
    sensors_scans["back"], sensors_angles["back"], sensors_poses["back"], alignment_axis="z"
)
coords_virtual = sensor_utils.transform_scan_to_cartesian(
    sensors_scans["virtual"], sensors_angles["virtual"], sensors_poses["virtual"], alignment_axis="z"
)
coords_fov_mask = sensor_utils.transform_scan_to_cartesian(
    ground_truth["fov_mask"], sensors_angles["virtual"], sensors_poses["virtual"], alignment_axis="z"
)

range_front = sensor_utils.cartesian_range(sensors_info["front"], sensors_poses["front"], alignment_axis="z")
range_back = sensor_utils.cartesian_range(sensors_info["back"], sensors_poses["back"], alignment_axis="z")
range_virtual = sensor_utils.cartesian_range(sensors_info["virtual"], sensors_poses["virtual"], alignment_axis="z")
range_kinect = sensor_utils.cartesian_range(sensors_info["kinect"], sensors_poses["kinect"], alignment_axis="y")

humans_arrows_gt = vis_utils.human_data_to_arrows(
    ground_truth["presence"],
    ground_truth["distance"],
    ground_truth["orientation_abs"],
    sensors_angles["virtual"],
    sensors_poses["virtual"],
)


plot_info = [
    # vis_utils.make_sensor_plot("front", "orange"),
    # vis_utils.make_sensor_plot("back", "red"),
    # vis_utils.make_sensor_plot("camera_fov_mask", "cyan"),
    vis_utils.make_sensor_plot("virtual", "green"),
    vis_utils.make_sensor_range_plot("kinect", "blue"),
    vis_utils.make_humans_plot("humans", "red"),
    vis_utils.make_humans_plot("humans_rel", "black"),
]

fig = go.Figure(
    data=plot_info,
    layout=dict(width=1280, height=720),
)

fig.update_xaxes(range=[-10, 10])
fig.update_yaxes(range=[-10, 10], scaleanchor="x", scaleratio=1)

for i in tqdm.trange(0, coords_front.size(0), desc="render", ncols=100):
    fig.update_traces(x=coords_front[i, :, 0], y=coords_front[i, :, 1], selector=dict(name="front"))
    fig.update_traces(x=coords_back[i, :, 0], y=coords_back[i, :, 1], selector=dict(name="back"))
    fig.update_traces(x=coords_virtual[i, :, 0], y=coords_virtual[i, :, 1], selector=dict(name="virtual"))

    fig.update_traces(x=coords_fov_mask[i, :, 0], y=coords_fov_mask[i, :, 1], selector=dict(name="camera_fov_mask"))
    fig.update_traces(x=range_kinect[i, :, 0], y=range_kinect[i, :, 1], selector=dict(name="kinect_range"))

    fig.update_traces(
        x=humans_arrows_gt[i, :, 0].flatten(), y=humans_arrows_gt[i, :, 1].flatten(), selector=dict(name="humans")
    )

    fig.write_image(tmp_folder / f"frame_{str(i).zfill(8)}.png")

_, exit_code = shell_utils.cmd(
    "ffmpeg",
    "-y",
    "-f",
    "image2",
    "-pattern_type",
    "glob",
    "-i",
    f"{tmp_folder}/*.png",
    "-framerate",
    "10",
    "-vcodec",
    "libx264",
    f"{output_path}",
)

if not exit_code:
    shell_utils.cmd("rm", "-r", str(tmp_folder))
else:
    raise ValueError("Could not generate video.")

fig.write_html(output_path.with_suffix(".html"))
