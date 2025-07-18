# 🌊 Flooded Areas AI

This repository contains the source code for my Master's Thesis titled "Assessment of neural networks for semantic segmentation of flooded areas".

In recent years, there has been an extraordinary development in the field of artificial intelligence, particularly in machine learning and neural networks. Simultaneously, modern climate change, the intensification of extreme weather events, and the growing population living in vulnerable areas make the problem of floods increasingly relevant and in need of effective monitoring and management tools.

In the face of these challenges, it is crucial to develop technologies that enable efficient monitoring and analysis of flood-prone areas. One of the promising directions in this field is the use of artificial intelligence, especially neural networks, for the automatic identification of flooded areas. This allows for fast and precise detection and classification of vulnerable areas, which is key for decision-making during crisis situations.

## 📊 Dataset
The project uses the [FloodNet](https://github.com/BinaLab/FloodNet-Challenge-EARTHVISION2021) dataset (not included in this repository), a curated collection of real-world aerial images for flood scene understanding and analysis. It is designed to support both image classification and semantic segmentation tasks in the context of flood disaster response.

## 📁 Project Structure
```yaml
├── configs/ # YAML config files for model training
│ ├── classification/ # Configs for classification models
│ └── segmentation/ # Configs for segmentation models
│
├── dataset/ # Contains dataset root foler, metadata and class mappings
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

## Models

## Installation

## Execution

#### Run program:
``` python3 src/main.py -c 0 -v -l ```
