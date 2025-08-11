# 🌊 Flooded Areas AI

This repository contains the source code for my Master's Thesis titled "Assessment of neural networks for semantic segmentation of flooded areas".

In recent years, there has been an extraordinary development in the field of artificial intelligence, particularly in machine learning and neural networks. Simultaneously, modern climate change, the intensification of extreme weather events, and the growing population living in vulnerable areas make the problem of floods increasingly relevant and in need of effective monitoring and management tools.

In the face of these challenges, it is crucial to develop technologies that enable efficient monitoring and analysis of flood-prone areas. One of the promising directions in this field is the use of artificial intelligence, especially neural networks, for the automatic identification of flooded areas. This allows for fast and precise detection and classification of vulnerable areas, which is key for decision-making during crisis situations.

## 📊 Dataset
The project uses the [FloodNet](https://github.com/BinaLab/FloodNet-Challenge-EARTHVISION2021) dataset (not included in this repository), a curated collection of real-world aerial images for flood scene understanding and analysis. It is designed to support both image classification and semantic segmentation tasks in the context of flood disaster response.

Classes in the dataset:
```json
{
    "0": "Background",
    "1": "Building-flooded",
    "2": "Building-non-flooded",
    "3": "Road-flooded",
    "4": "Road-non-flooded",
    "5": "Water",
    "6": "Tree",
    "7": "Vehicle",
    "8": "Pool",
    "9": "Grass"
}
```

### Remarks
The dataset is not perfect. While working on the project, I discovered that image *7606* from the training set was faulty. The dataset distinguishes between 10 different classes of objects (indices 0-9), yet this label contained pixels marked with indices 0-11, thus being meaningless. In order to continue work, this image-label pair was deleted from the dataset.

## 📁 Project Structure
```yaml
├── configs/ # YAML config files for model training
│ ├── classification/ # Configs for classification models
│ └── segmentation/ # Configs for segmentation models
│
├── dataset/ # Contains dataset root foler, CSV classification data, class mappings
│
├── src/ # Main source code
│ ├── data/ # Dataset class definitions, miscellaneous data related items
│ ├── dataloaders/ # PyTorch dataloader definitions
│ ├── losses/ # Custom or predefined loss functions
│ ├── metrics/ # Evaluation metrics
│ ├── models/ # Model architectures (predefined or custom models, common abstraction interface)
│ ├── optimizers/ # Optimizer configurations
│ ├── utils/ # Utility functions (e.g., logging, environment setup, image transforms etc.)
│ ├── testing.ipynb # Notebook for evaluating/test runs
│ ├── visualise.ipynb # Notebook for visualizing outputs or data
│ └── main.py # Program entry point
│
├── .gitignore # Git ignored files and folders
└── README.md # Project overview and usage instructions
```

## Configuration
The project is based on YAML configuration files. They allow for easy customisation of the machine learning task being tested, models, evalution metrics collected and more. Below is a sample configuration file:
```yaml
task: segmentation # Type of ML task
model: # Model choice and related parameters
  name: deeplabv3
  params:
    num_classes: 10 # number of classes in the task
    pretrained: true # whether the model is pretrained or not, model specific
loss: cross_entropy # Loss function
optimizer: # Optimizer
  name: adam
  lr: 0.001
metrics: # Evalutation metrics to be collected
  accuracy:
    task: multiclass
    num_classes: 10
  iou:
    task: multiclass
    num_classes: 10
    average: none
batch_size: 32
train: # Training data and number of epochs
  epochs: 30
  inputs: dataset/FloodNet_dataset/train/image
  targets: dataset/FloodNet_dataset/train/label
val: # Validation data
  inputs: dataset/FloodNet_dataset/val/image
  targets: dataset/FloodNet_dataset/val/label
test: # Test data
  inputs: dataset/FloodNet_dataset/test/image
  targets: dataset/FloodNet_dataset/test/label
```

Configuration files are located in the `configs/` folder. You can use the preexisting ones, modify them or create your own, following the defined structure.

### Currently supported config parameters
- task: classification/segmentation
- model:
    - name: inception/resnet50/xception/deeplabv3/enet/pspnet/unet3plus
    - num_classes




## Models

## Installation
If you want to try this project for yourself, follow these steps to get started:

0. System requirements:
    - your preferred Linux distro
    - Python 3.8.10
    - a GPU is highly recommended
1. Clone this repository to your desired location.
    ```
    git clone https://github.com/marcelbieniek/flooded-areas-ai-pytorch.git
    ```
2. Create a virtual environment.
    ```
    python3 -m venv .venv
    ```
3. Activate the virtual environment.
    ```
    source .venv/bin/activate
    ```
4. Install dependencies from `requirements.txt`.
    ```
    pip install -r requirements.txt
    ```

## Usage
```
usage: python3 src/main.py [-h] [-c CUDA_DEVICE] [-v] [-r RUN] [-l]

An evaluation workflow for classification and semantic segmentation neural network models.

optional arguments:
  -h, --help            show this help message and exit
  -c CUDA_DEVICE, --cuda-device CUDA_DEVICE
                        Index of CUDA device to compute on, if available (default=0).
  -v, --verbose         Print additional information during execution.
  -r RUN, --run RUN     Specify which model configuration to run. All configuration files are expected to be in 'config' directory, positioned at the root of the project. Allowed values are:
                        - all -- Run all configs found in the 'config' directory.
                        - file_name -- Path to YAML file to use as config (requires file extension .yaml or .yml; path should begin from first level inside 'config' directory).
                        - subdir_name -- Name of subdirectory inside the 'config' directory. All configs from this and further subdirectories will be run (can be used for grouping configs eg. run all segmentation models).
  -l, --logs            Collect logs and save models to files.
```
