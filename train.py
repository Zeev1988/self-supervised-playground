from self_supervised_3d_tasks.algorithms import cpc, jigsaw, relative_patch_location, rotation, exemplar
from self_supervised_3d_tasks.utils.model_utils import init, print_flat_summary
from self_supervised_3d_tasks.utils.model_utils import apply_encoder_model_3d
from self_supervised_3d_tasks.utils.model_utils import get_writing_path
from pathlib import Path

import tensorflow.keras as keras
from self_supervised_3d_tasks.data.numpy_3d_loader import DataGeneratorUnlabeled3D, PatchDataGeneratorUnlabeled3D

from self_supervised_3d_tasks.data.make_data_generator import get_data_generators
from self_supervised_3d_tasks.data.image_2d_loader import DataGeneratorUnlabeled2D
from self_supervised_3d_tasks.data.numpy_2d_loader import Numpy2DLoader
import numpy as np
import json
import os
import glob
import shutil
import json

os.environ["CUDA_VISIBLE_DEVICES"] ="0"

keras_algorithm_list = {
    "cpc": cpc,
    "jigsaw": jigsaw,
    "rpl": relative_patch_location,
    "rotation": rotation,
    "exemplar": exemplar
}

keras_model_list = {
    "cpc": None,
    "jigsaw": None,
    "rpl": None,
    "rotation": None,
    "exemplar": None
}

keras_epoch_counter = {
    "cpc": 0,
    "jigsaw": 0,
    "rpl": 0,
    "rotation": 0,
    "exemplar": 0
}

data_gen_list = {
    "kaggle_retina": DataGeneratorUnlabeled2D,
    "pancreas3d": DataGeneratorUnlabeled3D,
    "pancreas2d": Numpy2DLoader,
    "brats": DataGeneratorUnlabeled3D,
    "ukb2d": DataGeneratorUnlabeled2D,
    "ukb3d": PatchDataGeneratorUnlabeled3D
}


def get_dataset(data_dir, batch_size, f_train, f_val, train_val_split, dataset_name,
                train_data_generator_args={}, val_data_generator_args={}, **kwargs):
    data_gen_type = data_gen_list[dataset_name]

    train_data, validation_data = get_data_generators(data_dir, train_split=train_val_split,
                                                      train_data_generator_args={**{"batch_size": batch_size,
                                                                                    "pre_proc_func": f_train},
                                                                                 **train_data_generator_args},
                                                      val_data_generator_args={**{"batch_size": batch_size,
                                                                                  "pre_proc_func": f_val},
                                                                               **val_data_generator_args},
                                                      data_generator=data_gen_type)

    return train_data, validation_data

def get_models_names(**kwargs):
    if kwargs['task'] == 'all':
        return list(keras_algorithm_list.keys())
    else:
        return [kwargs['task']]

def init_net(working_dir, **kwargs):
    img_shape = (32, 32, 32, 4)
    args_dir = kwargs['config_dir']
    enc_model,_ = apply_encoder_model_3d(img_shape, **kwargs)
    algorithm_list = get_models_names(**kwargs)
    for algo_key in algorithm_list:
        kwargs = json.loads(open(args_dir+algo_key+'_3d_brats.json', "r").read())
        keras_algorithm_list[algo_key] = keras_algorithm_list[algo_key].create_instance(**kwargs)
        keras_algorithm_list[algo_key].enc_model = enc_model
        keras_model_list[algo_key] = keras_algorithm_list[algo_key].get_training_model()
        print_flat_summary(keras_model_list[algo_key])

def post_fit(working_dir, algo_key, **kwargs):
    wights_file_name = glob.glob(str(working_dir)+"/weights-improvement-"+"*.hdf5")[0]
    keras_model_list[algo_key].load_weights(wights_file_name)
    enc_model = keras_algorithm_list[algo_key].get_encoder_from_model(keras_model_list[algo_key])

    keras_epoch_counter[algo_key] += 10
    algorithm_list = get_models_names(**kwargs)
    for algo_key in algorithm_list:
        keras_model_list[algo_key] = keras_algorithm_list[algo_key].set_encoder(keras_model_list[algo_key], enc_model)

    shutil.rmtree(str(working_dir))

def save_models(working_dir, **kwargs):
    os.mkdir(str(working_dir))
    algorithm_list = get_models_names(**kwargs)
    for algo_key in algorithm_list:
        keras_model_list[algo_key].save_weights(str(working_dir)+'/'+algo_key+"_weights"+".hdf5")
        keras_model_list[algo_key].save(str(working_dir) + '/' + algo_key + "_model" + ".hdf5")

    j = json.dumps(keras_epoch_counter)
    f = open(str(working_dir)+'/'+"epoch_counts.json", "w")
    f.write(j)
    f.close()

def train_model(algorithm, data_dir, dataset_name, root_config_file, epochs=250, batch_size=2, train_val_split=0.9,
                base_workspace="~/netstore/workspace/", save_checkpoint_every_n_epochs=5, **kwargs):
    kwargs["root_config_file"] = root_config_file
    args_dir = kwargs['config_dir']
    global_epochs = kwargs['global_epochs']
    local_epochs = kwargs['local_epochs']
    working_dir = get_writing_path(Path(base_workspace).expanduser() / (algorithm + "_" + dataset_name),
                                   root_config_file)
    callbacks = init_net(working_dir, **kwargs)

    algorithm_list = get_models_names(**kwargs)
    if (len(algorithm_list)) == 1:
        local_epochs = global_epochs
        global_epochs = 1

    for i in np.arange(global_epochs):
      algo_key = np.random.choice(algorithm_list)
      model = keras_model_list[algo_key]
      f_train, f_val = keras_algorithm_list[algo_key].get_training_preprocessing()
      args = json.loads(open(args_dir+algo_key+'_3d_brats.json', "r").read())
      train_data, validation_data = get_dataset(f_train=f_train, f_val=f_val, train_val_split=train_val_split, **args)
      print("epoch "+ str(i)+": "+ algo_key)

      tb_c = keras.callbacks.TensorBoard(log_dir=str(working_dir))
      mc_c = keras.callbacks.ModelCheckpoint(str(working_dir / "weights-improvement-{epoch:03d}.hdf5"),
                                             monitor="val_loss",
                                             mode="min", save_best_only=True)  # reduce storage space
      mc_c_epochs = keras.callbacks.ModelCheckpoint(str(working_dir / "weights-{epoch:03d}.hdf5"),
                                                    period=1)  # reduce storage space
      callbacks = [tb_c, mc_c, mc_c_epochs]

      # Trains the model
      model.fit_generator(
          generator=train_data,
          steps_per_epoch=len(train_data),
          validation_data=validation_data,
          validation_steps=len(validation_data),
          epochs=local_epochs,
          callbacks=callbacks
      )
      if (len(algorithm_list)) != 1:
        post_fit(working_dir, algo_key, **kwargs)

    if (len(algorithm_list)) != 1:
      save_models(working_dir,**kwargs)

def main():
    init(train_model)


if __name__ == "__main__":
    main()
