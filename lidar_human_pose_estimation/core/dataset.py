import h5py
import torch
import pathlib

import torch.utils
import torch.utils.data
import numpy as np
from typing import List, Optional
from lidar_human_pose_estimation.core.config import Batch, Transform
from lidar_human_pose_estimation.utils import aug_utils
from lidar_human_pose_estimation.utils import temporal_registration_utils


class H5Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filename: pathlib.Path,
        group: str = "/",
        keys: Optional[List[str]] = None,
        transform: Optional[Transform] = None,
        history_parameters: dict = {},
        optitrack_filter = False,
    ) -> None:
        self.group = group
        self.filename = filename
        self.history_parameters = history_parameters

        if transform is None:
            transform = lambda x: x  # noqa: E731
        self.transform = transform

        self.h5f = h5py.File(filename, "r", libver="latest")[group]

        if keys is None:
            keys = list(self.h5f.keys())
        self.keys = keys


        if optitrack_filter:
            self.indices = np.arange(self.h5f['scan_virtual_history'].shape[0])
        else:
            self.indices = np.arange(self.h5f['scan_virtual_history'].shape[0])

        self.optitrack = optitrack_filter
    
    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Batch:
        dataset_batch = {k: self.h5f[k][self.indices[index]] for k in self.keys}
        history_key = list(filter(lambda x : 'history' in x, dataset_batch.keys()))[0]
        dataset_batch[history_key] = temporal_registration_utils.subsample_history(
            dataset_batch[history_key], self.history_parameters
        )

        if self.optitrack:
            dataset_batch['humans_presence_sensor'] = dataset_batch['humans_presence_optitrack']
            dataset_batch['humans_relative_bearing_sensor'] = dataset_batch['humans_relative_bearing_optitrack']
            dataset_batch['humans_distance_sensor'] = dataset_batch['humans_distance_optitrack']
            dataset_batch['scan_virtual_history'] = dataset_batch['scan_virtual_history']
            dataset_batch['camera_fov_mask'] = np.ones_like(dataset_batch['humans_distance_sensor'])
        return self.transform(dataset_batch)

def get_dataset(datasets_filenames: List[pathlib.Path], augment: bool = False, history_parameters=dict,
                split = 'training') -> H5Dataset:
    train_keys = [
        "scan_virtual_history",
        "camera_fov_mask",
        "humans_presence_sensor",
        "humans_relative_bearing_sensor",
        "humans_distance_sensor",
    ]

    test_keys = [
        "scan_virtual_history",
        "humans_presence_optitrack",
        "humans_relative_bearing_optitrack",
        "humans_distance_optitrack",
    ]

    if split == 'training': 
        keys = train_keys
    elif split == 'testing':
        keys = test_keys

    datasets = []
    for datasets_filename in datasets_filenames:
        dataset = H5Dataset(
            filename=datasets_filename,
            transform=aug_utils.get_transform(augment),
            history_parameters=history_parameters,
            keys=keys,
            optitrack_filter = split == 'testing'
        )
        datasets.append(dataset)

    return torch.utils.data.ConcatDataset(datasets)


if __name__ == "__main__":
    from pprint import pprint
    from config import parse_args
    from typing import Any, List, Dict, Callable, Union

    args = parse_args("input")

    dataset = get_dataset(
        filename=args.input,
        augment=True,
    )

    def apply_if_array(x: Any, fna: Callable[[Union[np.ndarray, torch.Tensor]], Any], fnb: Callable[[Any], Any]) -> Any:
        if isinstance(x, np.ndarray) or isinstance(x, torch.Tensor):
            return fna(x)
        else:
            return fnb(x)

    def desc(batch: Batch) -> Dict[str, str]:
        return {
            k: f"{apply_if_array(v, lambda x: x.shape, len)} {apply_if_array(v, lambda x: x.dtype, type)}"
            for k, v in batch.items()
        }

    print("Dataset length:", len(dataset))
    pprint(desc(dataset[0]))
