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

To train any of the self-supervised tasks with a specific algorithm, run: <br>
```train.py self_supervised_3d_tasks/configs/train/{algorithm}_{dimension}.json```

To run the downstream task and initialize the weights from a pretrained checkpoint, run: <br>
```finetune.py self_supervised_3d_tasks/configs/finetune/{algorithm}_{dimension}.json```
