# self-supervised-playground
based on https://github.com/HealthML/self-supervised-3d-tasks.git

## Usage instructions
In this codebase we provide configurations for training/evaluation of our models.

## Preparing data
**Our implementations of the algorithms require the data to be squared for 3D cubic.**

## Clone the repository and install dependencies
```
git clone https://github.com/Zeev1988/self-supervised-playground.git
cd self-supervised-playground
conda env create -f env_all_platforms.yml
conda activate conda-env
pip install -e .
```

## Running the experiments

To train any of the self-supervised tasks with a specific task, run: <br>
```train.py configs/train/base_3d_brats.json```

In base_3d_brats.json choose which task you want to train by updating the 'task' field with the relevant task name.<br>
to run the combination of the tasks set it to 'all'.<br>

To run the downstream task and initialize the weights from a pretrained checkpoint( for now only support cpc), run: <br>
```finetune.py configs/finetune/cpc_3d_brats.json```
