import torch
import pathlib
import argparse
import numpy as np
from numbers import Number
from datetime import datetime
from typing import Dict, Callable, List, Union
import importlib.resources

_eps = 1e-7
_d_max = 10.0
_config_terms = ["help", "input", "output", "model", "train", "multiple_input", "test", "grid_train", "filter"]
_package_folder = importlib.resources.files("lidar_human_pose_estimation").parent  # Base folder of general package

Sensor = Dict[str, Number]
Batch = Dict[str, torch.Tensor]
Transform = Callable[[Batch], Batch]
ArrayLike = Union[List[List[Number]], np.ndarray, torch.Tensor]


def parse_args(*config: str, print_args: bool = True) -> argparse.Namespace:
    """Parse command line aguments."""
    parser = argparse.ArgumentParser()

    for c in config:
        if c not in _config_terms:
            raise ValueError(
                f'Configuration argument "{c}" not recognized. Supported options are: {", ".join(_config_terms)}.'
            )

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="when set, avoids printing to terminal",
    )

    if "input" in config:
        parser.add_argument(
            "-i",
            "--input",
            type=pathlib.Path,
            required=True,
            help="filename of the input",
        )

    if "multiple_input" in config:
        parser.add_argument(
            "-i",
            "--multiple_input",
            type=pathlib.Path,
            required=False,
            help="filenames of the inputs",
            nargs="+",
        )

    if "output" in config:
        parser.add_argument(
            "-o",
            "--output",
            type=pathlib.Path,
            required=True,
            help="filename of the output",
        )

    if "filter" in config:
        parser.add_argument(
            "-f",
            "--filter",
            type=bool,
            required=False,
            help="whether to filter the data",
            default=False,
        )

    if any(c in config for c in ["model", "train"]):
        parser.add_argument(
            "-d",
            "--device",
            type=str,
            help=argparse.SUPPRESS,
            default="cuda" if torch.cuda.is_available() else "cpu",
        )
        parser.add_argument(
            "-m",
            "--model",
            type=pathlib.Path,
            help="filename of the model",
            default="./model/model_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
        parser.add_argument(
            "-bs",
            "--batch-size",
            type=int,
            help="size of the batches of data",
            default=64,
        )

    if "train" in config:
        parser.add_argument(
            "-e",
            "--epochs",
            type=int,
            help="number of epochs of the training phase",
            default=100,
        )
        parser.add_argument(
            "-lr",
            "--learning-rate",
            type=float,
            help="learning rate used for the training phase",
            default=1e-3,
        )
        parser.add_argument(
            "-v", "--validation", type=pathlib.Path, help="filename of the validation set", required=False, nargs="+"
        )

    if "help" in config:
        parser.print_help()
        exit()

    if "grid_train" in config:
        parser.add_argument(
            "-d",
            "--device",
            nargs="+",
            type=str,
            required=True,
        )

        parser.add_argument("--python-alias", default="python3", type=str, required=False)

        parser.add_argument("-j", "--jobs-per-device", type=int, default=1)

    args = parser.parse_args()

    if print_args and not args.quiet:
        for k, v in vars(args).items():
            print(f'{k:<20} = "{v}"')

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, choices=_config_terms, nargs="+")
    args = parser.parse_args()

    parse_args(*args.config, "help")
