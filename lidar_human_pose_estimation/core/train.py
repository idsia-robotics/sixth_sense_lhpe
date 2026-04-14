import os
from numbers import Number
from pprint import pprint
import shutil
import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Dict
import yaml
from lidar_human_pose_estimation.core.model import FCN, LHPELossFunction
from lidar_human_pose_estimation.core.dummy_model import DummyModel
from lidar_human_pose_estimation.core.config import parse_args, _package_folder
from lidar_human_pose_estimation.core.dataset import get_dataset
from lidar_human_pose_estimation.core import run
from lidar_human_pose_estimation.utils import naming_utils


def dict2mdtable(d: Dict[str, Number], key: str = "Name", val: str = "Value") -> str:
    rows = [f"| {key} | {val} |"]
    rows += ["|--|--|"]
    rows += [f"| {k} | {v} |" for k, v in d.items()]
    return "  \n".join(rows)


args = parse_args("train", "input")
# Get training config
with open(args.input, "r") as yaml_file:
    training_config = yaml.safe_load(yaml_file)

if training_config["model_nickname"]:
    log_path = checkpoint_path = naming_utils.create_model_name(training_config, args.model)
else:
    log_path = checkpoint_path = args.model

log_path = _package_folder / log_path

# Create save folder
if log_path.is_dir():
    raise FileExistsError(f"Model {log_path} already exist. Aborting!")

print(f"Saving model to {log_path}")
log_path.mkdir()
log_path.mkdir(parents=True, exist_ok=True)
checkpoint_path.mkdir(parents=True, exist_ok=True)


# Print history parameters
pprint(training_config["history_parameters"])
# Print model parameters
pprint(training_config["fcn_configs"])

# Train and validation datasets
train_files = training_config["training_files"]
train_files = [_package_folder / file for file in train_files]
val_files = training_config["validation_files"]
val_files = [_package_folder / file for file in val_files]

train_dataset = get_dataset(
    datasets_filenames=train_files,
    augment=training_config["train_configs"]["augmentations"],
    history_parameters=training_config["history_parameters"],
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=training_config["train_configs"]["batch_size"],
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    pin_memory_device = args.device
)
print(f"Training on {len(train_dataset)} datapoints.")

val_dataset = get_dataset(
    datasets_filenames=val_files, augment=False, history_parameters=training_config["history_parameters"]
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=training_config["train_configs"]["batch_size"],
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    pin_memory_device = args.device
)
print(f"Validating on {len(val_dataset)} datapoints.")

writer = SummaryWriter(log_dir=log_path, flush_secs=30)
writer.add_text("train_args/all", dict2mdtable(vars(args)))
writer.add_text("train_config/all", dict2mdtable(training_config))

# Loss function
loss_function_params = training_config["train_configs"]["loss_function"]
loss_function = LHPELossFunction(
    presence=loss_function_params["presence"],
    distance=loss_function_params["distance"],
    orientation=loss_function_params["orientation"],
    bidirection=loss_function_params["bidirection"],
    verse=loss_function_params["verse"],
)

if training_config["train_configs"]["dummy_model"]:
    optimizer = None
    model = DummyModel(dataloader=train_loader, dummy_type=training_config["train_configs"]["dummy_type"])
else:
    model = FCN(
        input_channels=training_config["history_parameters"]["length"],
        use_skip_connection=training_config["fcn_configs"]["use_skip_connection"],
        layer_configs=training_config["fcn_configs"]["layer_configs"],
        loss_activation=training_config["train_configs"]["loss_function"],
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=training_config["train_configs"]["learning_rate"])

# Dumping training config
yaml.dump(training_config, open(os.path.join(log_path, "train_config.yaml"), "w"))

best_val_loss = float("inf")
for epoch in tqdm.trange(1, training_config["train_configs"]["num_epochs"] + 1, desc="epoch", ncols=100):
    model.train()
    train_metrics = run.run_epoch(
        model=model,
        dataloader=train_loader,
        args=args,
        optimizer=optimizer,
        selected_loss_function=loss_function.loss_function,
        training_config=training_config,
    )
    model.eval()
    with torch.no_grad():
        val_metrics = run.run_epoch(
            model=model,
            dataloader=val_loader,
            args=args,
            optimizer=None,
            selected_loss_function=loss_function.loss_function,
            training_config=training_config,
        )

    if optimizer is not None:
        lr = optimizer.param_groups[0]["lr"]
    else:
        lr = training_config["train_configs"]["learning_rate"]

    current_val_loss = val_metrics["loss"]
    if current_val_loss <= best_val_loss:
        best_val_loss = current_val_loss
        torch.save(model.state_dict(), log_path / "best.pth")

    torch.save(model.state_dict(), log_path / "last.pth")

    writer.add_scalar("train/lr", lr, epoch)

    for metric, value in train_metrics.items():
        writer.add_scalar(f"train/{metric}", train_metrics[metric], epoch)

    for metric, value in val_metrics.items():
        writer.add_scalar(f"val/{metric}", val_metrics[metric], epoch)

    writer.flush()
