import math
import tqdm
import torch
from lidar_human_pose_estimation.utils.geom_utils import detection_matching
import plotly.graph_objects as go
from PIL import Image
import yaml
from lidar_human_pose_estimation.utils import (
    shell_utils,
    sensor_utils,
    vis_utils,
    post_processing_utils,
    gt_utils,
    temporal_registration_utils,
    matching_utils,
)
from lidar_human_pose_estimation.core.model import FCN
from lidar_human_pose_estimation.core.config import parse_args

# Bug fix for Kaleido
import plotly.io as pio

pio.kaleido.scope.mathjax = None

args = parse_args("input", "model", "output")
input_path = args.input
output_path = args.output
model_path = args.model
device = args.device

tmp_folder = output_path.parent / "tmp"
extracted_frames_folder = output_path.parent / "extracted_frames"
head_rotated_folder = output_path.parent / "head_rotated"
# If tmp_folder exists delete it
if tmp_folder.exists():
    shell_utils.cmd("rm", "-r", str(tmp_folder))
tmp_folder.mkdir()

# Model parameters loading
with open(args.model / "train_config.yaml", "r") as yaml_file:
    training_config = yaml.safe_load(yaml_file)

model = FCN(
    input_channels=training_config["history_parameters"]["length"],
    use_skip_connection=training_config["fcn_configs"]["use_skip_connection"],
    layer_configs=training_config["fcn_configs"]["layer_configs"],
    loss_activation=training_config["train_configs"]["loss_function"],
).to(args.device)
model.load_state_dict(torch.load(args.model / "best.pth", map_location=args.device))
model.eval()

# Read data from h5 file
sensors_info, sensors_angles, sensors_scans, sensors_poses, gt_sensor = vis_utils.read_h5_file(input_path, device)
gt_optitrack = vis_utils.read_h5_file_optitrack(input_path, device)

# Resize history to what we need
sensors_scans["virtual_history"] = temporal_registration_utils.subsample_history(
    sensors_scans["virtual_history"], training_config["history_parameters"]
)
sensors_poses["virtual_history"] = torch.eye(4, dtype=torch.float64, device=device)[None, None].tile(
    sensors_scans["virtual_history"].shape[0], sensors_scans["virtual_history"].shape[1], 1, 1
)

# Prediction
with torch.no_grad():
    pred_raw = model(sensors_scans["virtual_history"].float())

nms_threshold = 0.90
nms = post_processing_utils.CircularTensorNMS(nms_threshold)
pred = nms.iterative_peak_nms(pred_raw)

pred["orientation_rel"] = torch.atan2(pred["sine"], pred["cosine"])
pred["orientation_abs"] = gt_utils.absolute_bearing(pred["orientation_rel"], sensors_angles["virtual"])
# Model Output Visualization
presence_vis_params = {"min": 0.55, "max": 1.5}
presence_vis_params["interval"] = presence_vis_params["max"] - presence_vis_params["min"]
pred["presence_vis_thresh"] = sensor_utils.transform_scan_to_cartesian(
    torch.ones_like(pred_raw["presence"]) * presence_vis_params["interval"] * nms_threshold
    + presence_vis_params["min"],
    sensors_angles["virtual"],
    sensors_poses["virtual"],
    alignment_axis="z",
)[:, ::5, :]

nms_mask = pred_raw["presence"] > nms_threshold
presence_vis = torch.where(~nms_mask, pred_raw["presence"], torch.nan)
presence_vis = vis_utils.substitute_limits_nans_with_value(presence_vis, nms_threshold)
pred["presence_vis"] = sensor_utils.transform_scan_to_cartesian(
    presence_vis * presence_vis_params["interval"] + presence_vis_params["min"],
    sensors_angles["virtual"],
    sensors_poses["virtual"],
    alignment_axis="z",
)
presence_nms_vis = torch.where(nms_mask, pred_raw["presence"], torch.nan)
presence_nms_vis = vis_utils.substitute_limits_nans_with_value(presence_nms_vis, nms_threshold)
pred["presence_nms_vis"] = sensor_utils.transform_scan_to_cartesian(
    presence_nms_vis * presence_vis_params["interval"] + presence_vis_params["min"],
    sensors_angles["virtual"],
    sensors_poses["virtual"],
    alignment_axis="z",
)

# Scatterplot
# vis_utils.model_prediction_scatter_plot(gt_sensor, pred_raw, output_path)

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

coords_virtual_history = sensor_utils.transform_scan_to_cartesian(
    sensors_scans["virtual_history"].reshape(-1, sensors_scans["virtual_history"].size(-1)),
    sensors_angles["virtual"],
    sensors_poses["virtual_history"].view(-1, 1, 4, 4),
    alignment_axis="z",
).view(
    sensors_scans["virtual_history"].shape[0],
    sensors_scans["virtual_history"].shape[1],
    sensors_scans["virtual_history"].shape[2],
    2,
)
coords_fov_mask = sensor_utils.transform_scan_to_cartesian(
    gt_sensor["fov_mask"], sensors_angles["virtual"], sensors_poses["virtual"], alignment_axis="z"
)
if coords_virtual_history.size(1) > 10:
    # Subsample history
    subsample = math.ceil(coords_virtual_history.size(1) // 10)
    coords_virtual_history = coords_virtual_history[:, ::subsample, :, :]

# range_front = sensor_utils.cartesian_range(sensors_info["front"], sensors_poses["front"], alignment_axis="z")
# range_back = sensor_utils.cartesian_range(sensors_info["back"], sensors_poses["back"], alignment_axis="z")
# range_virtual = sensor_utils.cartesian_range(sensors_info["virtual"], sensors_poses["virtual"], alignment_axis="z")`
range_kinect = sensor_utils.cartesian_range(sensors_info["kinect"], sensors_poses["kinect"], alignment_axis="y")

head_yaws = torch.atan2(sensors_poses["kinect"][:, :, 1, 2], sensors_poses["kinect"][:, :, 0, 2]).squeeze(-1)
head_yaws_degrees = torch.round(torch.rad2deg(head_yaws)).to(torch.int)

humans_arrows_gt_sensor = vis_utils.human_data_to_arrows(
    gt_sensor["presence"],
    gt_sensor["distance"],
    gt_sensor["orientation_abs"],
    sensors_angles["virtual"],
    sensors_poses["virtual"],
)
if gt_optitrack is not None:
    humans_arrows_gt_optitrack = vis_utils.human_data_to_arrows(
        gt_optitrack["presence"],
        gt_optitrack["distance"],
        gt_optitrack["orientation_abs"],
        sensors_angles["virtual"],
        sensors_poses["virtual"],
    )

humans_arrows_pred = vis_utils.human_data_to_arrows(
    pred["presence"],
    pred["distance"],
    pred["orientation_abs"],
    sensors_angles["virtual"],
    sensors_poses["virtual"],
)

# Detection matching
prediction_detections = matching_utils.data_to_detections(
    pred["presence"],
    pred["distance"],
    pred["orientation_abs"],
    sensors_angles["virtual"],
    sensors_poses["virtual"],
)

if gt_optitrack is not None:
    gt_optitrack_detections = matching_utils.data_to_detections(
        gt_optitrack["presence"],
        gt_optitrack["distance"],
        gt_optitrack["orientation_abs"],
        sensors_angles["virtual"],
        sensors_poses["virtual"],
    )

    print("Matching")
    matching_results = matching_utils.match_detections(
        detections_gt=gt_optitrack_detections,
        detections_pred=prediction_detections,
        max_cost=0.5,
    )

    metrics = matching_utils.compute_matching_metrics(matching_results)
    print(metrics)

mode_video_settings = {
    "plot_dimensions": (1920, 1080),
    "plot_range": {"x": [-6.25, 6.25], "y": [-6.0, 6.0]},
    "marker_size": {"humans": 10, "sensors": 6, "model": 2},
    "font": {"name": "Arial", "size": 20},
    "show_legend": True,
    "ticks": True,
    "scale_bar": False,
    "plot_background_color": "rgba(228,236,246,255)",
    "grid_color": "rgba(255,255,255,255)",
    "plot_history": False,
}
mode_plot_settings = {
    "plot_dimensions": (1080, 1080),
    "plot_range": {"x": [-6.25, 6.25], "y": [-6.0, 6.0]},
    "marker_size": {"humans": 20, "sensors": 12, "model": 4},
    "font": {"name": "Times New Roman", "size": 40},
    "show_legend": False,
    "ticks": False,
    "scale_bar": True,
    "plot_background_color": "rgba(250,250,250,255)",
    "grid_color": "rgba(215,215,215,215)",
    "plot_history": False,
}

mode = "video"
mode = "plot"
extract_frames = None
# extract_frames = [3550]

if mode == "plot":
    settings = mode_plot_settings
elif mode == "video":
    settings = mode_video_settings

# Change here to make single component appear 
plot_data = [
    # LiDARs
    # vis_utils.make_sensor_plot("Front Sensor", "orange"),
    # vis_utils.make_sensor_plot("Rear Sensor", "red"),
    vis_utils.make_sensor_plot("LiDAR", "green", settings["marker_size"]["sensors"]),
    # Azure Kinect
    # vis_utils.make_sensor_plot("camera_fov_mask", "cyan"),
    vis_utils.make_sensor_range_plot("Azure Kinect FOV", "red", alpha=0.2),
    # Ground truth
    # vis_utils.make_humans_plot("Motion Capture", "darkblue", 1.0, settings["marker_size"]["humans"]),
    vis_utils.make_humans_plot("Azure Kinect", "red", 1.0, settings["marker_size"]["humans"]),
    # # Model output
    vis_utils.make_model_range_plot("Detection Threshold", "darkgrey", 1.0, settings["marker_size"]["model"]),
    vis_utils.make_model_plot("Rejected", "black", 0.1, settings["marker_size"]["model"]),
    vis_utils.make_model_plot("Accepted", "black", 1.0, settings["marker_size"]["model"]),
    # Utils
    # vis_utils.make_humans_plot("timestamp", "black"),
]

if gt_optitrack is not None:
    # Predictions
    plot_data.append(
        vis_utils.make_humans_plot("Matched Predictions (TP)", "mediumorchid", 1.0, settings["marker_size"]["humans"]),
    )
    plot_data.append(
        vis_utils.make_humans_plot("Unmatched Predictions (FP)", "orange", 1.0, settings["marker_size"]["humans"]),
    )
    # plot_data.append(
    #     vis_utils.make_humans_plot("humans_predicted_kf", "purple"),
    # )
    pass
else:
    # plot_data.append(
    #     vis_utils.make_humans_plot("Predictions", "purple", 1.0, settings["marker_size"]["humans"]),
    # )
    pass

colors = {
    0: "darkgreen",
    1: "forestgreen",
    2: "seagreen",
    3: "mediumseagreen",
    4: "olivedrab",
    5: "limegreen",
    6: "springgreen",
    7: "lawngreen",
    8: "lightgreen",
    9: "honeydew",
}

virtual_history_fields_names = [
    "History_-" + str(i * training_config["history_parameters"]["stride"])
    for i in range(coords_virtual_history.size(1))
]

plot_order = torch.arange(0, len(virtual_history_fields_names)).tolist()
plot_order.reverse()

if settings["plot_history"]:
    for j in plot_order:
        plot_data.append(vis_utils.make_sensor_plot(virtual_history_fields_names[j], colors[j % len(colors)]))


fig = go.Figure(
    data=plot_data,
    layout=dict(width=settings["plot_dimensions"][0], height=settings["plot_dimensions"][1]),
)

fig.update_xaxes(
    title="Distance [m]",
    range=settings["plot_range"]["x"],
    dtick=2,
    gridcolor=settings["grid_color"],
    zerolinecolor=settings["grid_color"],
)
fig.update_yaxes(
    title="Distance [m]",
    range=settings["plot_range"]["y"],
    dtick=2,
    gridcolor=settings["grid_color"],
    zerolinecolor=settings["grid_color"],
    scaleanchor="x",
    scaleratio=1,
)
if not settings["ticks"]:
    fig.update_layout(
        xaxis=dict(showticklabels=False, title=None),  # Hide x-axis ticks
        yaxis=dict(showticklabels=False, title=None),  # Hide y-axis ticks
    )
fig.update_layout(plot_bgcolor=settings["plot_background_color"])  # Background of the plotting area

if mode == "video":
    fig.update_layout(
        legend=dict(bgcolor="rgba(0,0,0,0)", yanchor="top", y=0.99, xanchor="left", x=0.01, indentation=20),
        font=dict(size=settings["font"]["size"], family=settings["font"]["name"], color="black"),
    )
elif mode == "plot":
    fig.update_layout(
        legend=dict(
            orientation="h",
            entrywidth=(settings["plot_dimensions"][0] - 100) / 4.0,
            yanchor="bottom",
            x=0.0,
            y=1.0,
            xanchor="left",
        ),
        font=dict(size=settings["font"]["size"], family=settings["font"]["name"], color="black"),
        showlegend=settings["show_legend"],
    )

vis_utils.rotate_and_save_images(
    head_yaws_degrees,
    "/home/arsimone/ros_ws/tiago_devel_ws/src/tiago_social_hri/non_ros/lidar_human_pose_estimation/out/tiago_simplified_head.png",
    head_rotated_folder,
)

if settings["scale_bar"]:
    fig.add_shape(
        type="line",
        x0=-6.0,
        x1=-4.0,
        y0=-4.0,
        y1=-4.0,
        line=dict(color="black", width=2),
    )

    fig.add_shape(
        type="line",
        x0=-6.0,
        x1=-6.0,
        y0=-4.2,
        y1=-3.8,
        line=dict(color="black", width=2),
    )

    fig.add_shape(
        type="line",
        x0=-4.0,
        x1=-4.0,
        y0=-4.2,
        y1=-3.8,
        line=dict(color="black", width=2),
    )

    # Add a text annotation to label the scale bar
    fig.add_annotation(
        x=-5.1,
        y=-3.6,  # Adjust position
        text="2 m",
        showarrow=False,
        font=dict(size=40, color="black"),
    )

for i in tqdm.trange(0, coords_front.size(0), desc="render", ncols=100):
    if extract_frames is not None and i not in extract_frames:
        continue
    head_rotated_path = head_rotated_folder / f"image_{head_yaws_degrees[i]}.png"
    fig.update_layout(
        images=[
            dict(
                source="/home/arsimone/ros_ws/tiago_devel_ws/src/tiago_social_hri/non_ros/lidar_human_pose_estimation/out/tiago_simplified_body.png",  # Replace with the path to your PNG image
                x=0.5,  # x position of the image (0 to 1, relative to the plot)
                y=0.5,  # y position of the image (0 to 1, relative to the plot)
                xanchor="center",  # Anchor the image at its center
                yanchor="middle",  # Anchor the image at its middle
                sizex=0.08,  # Size of the image in x-direction (relative to the plot)
                sizey=0.08,  # Size of the image in y-direction (relative to the plot)
                opacity=1.0,  # Opacity of the image (0 to 1)
                layer="above",  # Place the image above the plot
            ),
            dict(
                source=str(head_rotated_path),  # Replace with the path to your PNG image
                x=0.51,  # x position of the image (0 to 1, relative to the plot)
                y=0.5,  # y position of the image (0 to 1, relative to the plot)
                xanchor="center",  # Anchor the image at its center
                yanchor="middle",  # Anchor the image at its middle
                sizex=0.08,  # Size of the image in x-direction (relative to the plot)
                sizey=0.08,  # Size of the image in y-direction (relative to the plot)
                opacity=1.0,  # Opacity of the image (0 to 1)
                layer="above",  # Place the image above the plot
            ),
        ]
    )

    # LiDARs
    fig.update_traces(x=coords_front[i, :, 0], y=coords_front[i, :, 1], selector=dict(name="Front Sensor"))
    fig.update_traces(x=coords_back[i, :, 0], y=coords_back[i, :, 1], selector=dict(name="Rear Sensor"))
    fig.update_traces(x=coords_virtual[i, :, 0], y=coords_virtual[i, :, 1], selector=dict(name="LiDAR"))

    # History
    if settings["plot_history"]:
        for j in plot_order:
            fig.update_traces(
                x=coords_virtual_history[i, j, :, 0],
                y=coords_virtual_history[i, j, :, 1],
                selector=dict(name=virtual_history_fields_names[j]),
            )

    # Camera
    fig.update_traces(x=coords_fov_mask[i, :, 0], y=coords_fov_mask[i, :, 1], selector=dict(name="camera_fov_mask"))
    fig.update_traces(x=range_kinect[i, :, 0], y=range_kinect[i, :, 1], selector=dict(name="Azure Kinect FOV"))

    # Humans GT
    if gt_optitrack is not None:
        fig.update_traces(
            x=humans_arrows_gt_optitrack[i, :, 0].flatten(),
            y=humans_arrows_gt_optitrack[i, :, 1].flatten(),
            selector=dict(name="Motion Capture"),
        )
    fig.update_traces(
        x=humans_arrows_gt_sensor[i, :, 0].flatten(),
        y=humans_arrows_gt_sensor[i, :, 1].flatten(),
        selector=dict(name="Azure Kinect"),
    )

    # Prediction
    if gt_optitrack is not None:
        mgts = matching_results[i]["gt_TP_ids"]
        mpreds = matching_results[i]["pred_TP_ids"]
        not_mpreds = matching_results[i]["pred_FP_ids"]

        humans_arrows_pred_mpred = humans_arrows_pred[i, mpreds]
        humans_arrows_pred_mpred = torch.nn.functional.pad(
            humans_arrows_pred_mpred,
            (0, 0, 0, 0, 0, humans_arrows_pred.shape[1] - humans_arrows_pred_mpred.shape[1]),
            "constant",
            torch.nan,
        )
        humans_arrows_pred_not_mpred = humans_arrows_pred[i, not_mpreds]
        humans_arrows_pred_not_mpred = torch.nn.functional.pad(
            humans_arrows_pred_not_mpred,
            (0, 0, 0, 0, 0, humans_arrows_pred.shape[1] - humans_arrows_pred_not_mpred.shape[1]),
            "constant",
            torch.nan,
        )

        fig.update_traces(
            x=humans_arrows_pred_mpred[:, 0].flatten(),
            y=humans_arrows_pred_mpred[:, 1].flatten(),
            selector=dict(name="Matched Predictions (TP)"),
        )

        fig.update_traces(
            x=humans_arrows_pred_not_mpred[:, 0].flatten(),
            y=humans_arrows_pred_not_mpred[:, 1].flatten(),
            selector=dict(name="Unmatched Predictions (FP)"),
        )
    else:
        fig.update_traces(
            x=humans_arrows_pred[i, :, 0].flatten(),
            y=humans_arrows_pred[i, :, 1].flatten(),
            selector=dict(name="Predictions"),
        )

    # Model output
    fig.update_traces(
        x=pred["presence_vis_thresh"][i, :, 0],
        y=pred["presence_vis_thresh"][i, :, 1],
        selector=dict(name="Detection Threshold"),
    )
    fig.update_traces(
        x=pred["presence_vis"][i, :, 0],
        y=pred["presence_vis"][i, :, 1],
        selector=dict(name="Rejected"),
    )
    fig.update_traces(
        x=pred["presence_nms_vis"][i, :, 0],
        y=pred["presence_nms_vis"][i, :, 1],
        selector=dict(name="Accepted"),
    )

    # Kalman Filter
    # people_detections = vis_utils.arrows_to_human_detections(humans_arrows_pred[i, :, :, :2])
    # # print(f"{i}")
    # people_tracker.update(people_detections)
    # people_tracked_dict = people_tracker.get_tracks_poses()
    # people_tracked_list = list(people_tracked_dict.values())
    # people_tracked_arrows = vis_utils.human_detections_to_arrows(people_tracked_list, shape=humans_arrows_pred[i].shape)

    # fig.update_traces(
    #     x=people_tracked_arrows[:, 0].flatten(),
    #     y=people_tracked_arrows[:, 1].flatten(),
    #     selector=dict(name="humans_predicted_kf"),
    # )

    fig.update_traces(
        x=torch.ones(1) * (settings["plot_range"]["x"][0] + 0.5),
        y=torch.ones(1) * (settings["plot_range"]["y"][0] + 0.5),
        mode="text",
        text=f"{i}",
        textposition="top center",
        selector=dict(name="timestamp"),
        textfont=dict(color="black", size=40),
    )

    if extract_frames is None:
        fig.write_image(tmp_folder / f"frame_{str(i).zfill(8)}.png")
    else:
        fig.write_image(extracted_frames_folder / f"frame_{str(i).zfill(8)}.pdf")

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
    f"{args.output}",
)

if not exit_code:
    shell_utils.cmd("rm", "-r", str(tmp_folder))
    shell_utils.cmd("rm", "-r", str(head_rotated_folder))
else:
    raise ValueError("Could not generate video.")

# fig.write_html(args.output.with_suffix(".html"))
