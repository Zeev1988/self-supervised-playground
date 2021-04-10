# self-supervised-playground
based on https://github.com/HealthML/self-supervised-3d-tasks.git
Overview

This codebase contains a implementation of five self-supervised representation learning techniques, utility code for running training and evaluation loops.
Usage instructions

In this codebase we provide configurations for training/evaluation of our models.

For debugging or running small experiments we support training and evaluation using a single GPU device.
Preparing data

Our implementations of the algorithms require the data to be squared for 2D or cubic for 3D images.
Clone the repository and install dependencies

Make sure you have anaconda installed.

Then perform the following commands, while you are in your desired workspace directory:

```
git clone https://github.com/HealthML/self-supervised-3d-tasks.git
cd self-supervised-playground
conda env create -f env_all_platforms.yml
conda activate conda-env
pip install -e .
```

Running the experiments

To train any of the self-supervised tasks with a specific algorithm, run python train.py self_supervised_3d_tasks/configs/train/{algorithm}_{dimension}.json To run the downstream task and initialize the weights from a pretrained checkpoint, run python finetune.py self_supervised_3d_tasks/configs/finetune/{algorithm}_{dimension}.json
