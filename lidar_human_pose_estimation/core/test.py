import pickle
import torch
import yaml
from lidar_human_pose_estimation.core.model import FCN
import lidar_human_pose_estimation.core.run as run
from lidar_human_pose_estimation.core.config import parse_args
from lidar_human_pose_estimation.core.dataset import get_dataset
from lidar_human_pose_estimation.utils.vis_utils import model_prediction_scatter_plot

args = parse_args("train")
with open(args.model / "train_config.yaml", "r") as yaml_file:
    training_config = yaml.safe_load(yaml_file)


dataset = get_dataset(
    datasets_filenames=args.validation,
    augment=False,
    history_parameters=training_config["history_parameters"],
    split='testing'
)

print(f"Testing on {len(dataset)} data points")
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=8,
    pin_memory='cuda' in args.device,
    pin_memory_device = args.device if 'cuda' in args.device else ''
)

model = FCN(
    input_channels=training_config["history_parameters"]["length"],
    use_skip_connection=training_config["fcn_configs"]["use_skip_connection"],
    layer_configs=training_config["fcn_configs"]["layer_configs"],
    loss_activation=training_config["train_configs"]["loss_function"],
).to(args.device)
model.load_state_dict(torch.load(args.model / "best.pth", map_location=args.device))
model.eval()

with torch.no_grad():
    metrics, inference_data, errors_per_thr, predictions_per_thr = run.run_epoch_testing(
        model=model,
        dataloader=dataloader,
        args=args,
        training_config=training_config,
    )

test_data = {
    'metrics' : metrics,
    'dataset' : [v.name for v in args.validation],
    'checkpoint' : 'best.pth'
}

model_prediction_scatter_plot(
    ground_truth = {
        'presence' : inference_data['true_presence'],
        'orientation_rel' : inference_data['true_ori'],
        'distance' : inference_data['true_dist'],
    },

    pred = {
        'distance' : inference_data['pred_dist'],
        'presence' : inference_data['pred_presence'],
        'orientation' : inference_data['pred_ori'],
        'sine' : inference_data['pred_sin'],
        'cosine' : inference_data['pred_cos'],
    },
    output=args.model / "performance_plot_fov.png"
)

with open(args.model / "test_performance_fov.yaml", "w") as f:
    yaml.safe_dump(test_data, f)

with open(args.model / "test_errors_per_thr_fov.pkl", "wb") as f:
    pickle.dump(errors_per_thr, f)

with open(args.model / "test_predictions_per_thr_fov.pkl", "wb") as f:
    pickle.dump(predictions_per_thr, f)
# print(*(f"{k:<20} = {v:.4f}" for k, v in metrics.items()), sep="\n")

