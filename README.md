# Sixth-Sense: Self-Supervised Learning of Spatial Awareness of Humans from a Planar Lidar

Simone Arreghini, Nicholas Carlotti, Mirko Nava, Antonio Paolillo, and Alessandro Giusti*

Dalle Molle Institute for Artificial Intelligence, USI-SUPSI, Lugano (Switzerland)

### Installation Instructions

To install this package, as well as any dependency:
```bash
pip install -e .
```
This will automatically install any dependency in your Python environment. Please use a version of Python>=3.10

### Usage

Download and extract our dataset from Zenodo at this [URL](https://zenodo.org/records/14936069).

Copy or make a symlink to the extracted content to the folder /hdf5 such that it directly contains the *break_area*, *corridor*, and *lab* folders.

#### Train a new model
The file "lidar_human_pose_estimation/config/train_config.yaml" can be used to change the model architecture as well as the loss function used.

A new model can be trained using the command:
```
python -m lidar_human_pose_estimation.core.train -i <PATH_TO_THE_TRAIN_CONFIG_FILE>
```

The model will be saved in the model folder of this repository.

#### Test a model performance
The performance of a model on a specific dataset can be tested with:
```
python -m lidar_human_pose_estimation.core.test -m <PATH_TO_THE_MODEL_FOLDER> -v <PATH_TO_THE_TEST_DATASET> --device cpu
```

This function will output some model performance metrics as well as create plots. 

#### Visualization
Depending on what you need to visualize three different Python scripts can be used.

To visualize only the content of an h5 file:
```
python -m lidar_human_pose_estimation.visualization.vis_h5 -i <PATH_TO_THE_DATASET> -o <PATH_TO_THE_VIDEO_OUTPUT_FOLDER>
```

To visualize also the optitrack data in a dataset (only for "Lab" environment): 
```
python -m lidar_human_pose_estimation.visualization.vis_h5_optitrack -i <PATH_TO_THE_DATASET> -o <PATH_TO_THE_VIDEO_OUTPUT_FOLDER>
```

To visualize how a model performs : 
```
python -m lidar_human_pose_estimation.visualization.vis_model -i  <PATH_TO_THE_DATASET> -m <PATH_TO_THE_MODEL_FOLDER> --device cpu -o <PATH_TO_THE_VIDEO_OUTPUT_FOLDER>
```


