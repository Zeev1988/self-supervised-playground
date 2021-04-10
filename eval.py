import csv
import functools
import gc
import os
import random
from os.path import expanduser
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import CSVLogger

import self_supervised_3d_tasks.utils.metrics as metrics
from self_supervised_3d_tasks.utils.callbacks import TerminateOnNaN, NaNLossError, LogCSVWithStart
from self_supervised_3d_tasks.utils.metrics import weighted_sum_loss, jaccard_distance, \
    weighted_categorical_crossentropy, weighted_dice_coefficient, weighted_dice_coefficient_loss, \
    weighted_dice_coefficient_per_class, brats_wt_metric, brats_et_metric, brats_tc_metric
from self_supervised_3d_tasks.test_data_backend import CvDataKaggle, StandardDataLoader
from self_supervised_3d_tasks.train import (
    keras_algorithm_list,
)
from self_supervised_3d_tasks.utils.model_utils import (
    apply_prediction_model,
    get_writing_path,
    print_flat_summary)
from self_supervised_3d_tasks.utils.model_utils import init

os.environ["CUDA_VISIBLE_DEVICES"] ="3"

def get_score(score_name):
    if score_name == "qw_kappa":
        return metrics.score_kappa
    elif score_name == "bin_accuracy":
        return metrics.score_bin_acc
    elif score_name == "cat_accuracy":
        return metrics.score_cat_acc
    elif score_name == "dice":
        return metrics.score_dice
    elif score_name == "dice_pancreas_0":
        return functools.partial(metrics.score_dice_class, class_to_predict=0)
    elif score_name == "dice_pancreas_1":
        return functools.partial(metrics.score_dice_class, class_to_predict=1)
    elif score_name == "dice_pancreas_2":
        return functools.partial(metrics.score_dice_class, class_to_predict=2)
    elif score_name == "jaccard":
        return metrics.score_jaccard
    elif score_name == "qw_kappa_kaggle":
        return metrics.score_kappa_kaggle
    elif score_name == "cat_acc_kaggle":
        return metrics.score_cat_acc_kaggle
    elif score_name == "brats_wt":
        return metrics.brats_wt
    elif score_name == "brats_tc":
        return metrics.brats_tc
    elif score_name == "brats_et":
        return metrics.brats_et
    else:
        raise ValueError(f"score {score_name} not found")


def make_custom_metrics(metrics):
    metrics = list(metrics)

    if "weighted_dice_coefficient" in metrics:
        metrics.remove("weighted_dice_coefficient")
        metrics.append(weighted_dice_coefficient)
    if "brats_metrics" in metrics:
        metrics.remove("brats_metrics")
        metrics.append(brats_wt_metric)
        metrics.append(brats_tc_metric)
        metrics.append(brats_et_metric)
    if "weighted_dice_coefficient_per_class_pancreas" in metrics:
        metrics.remove("weighted_dice_coefficient_per_class_pancreas")

        def dice_class_0(y_true, y_pred):
            return weighted_dice_coefficient_per_class(y_true, y_pred, class_to_predict=0)

        def dice_class_1(y_true, y_pred):
            return weighted_dice_coefficient_per_class(y_true, y_pred, class_to_predict=1)

        def dice_class_2(y_true, y_pred):
            return weighted_dice_coefficient_per_class(y_true, y_pred, class_to_predict=2)

        metrics.append(dice_class_0)
        metrics.append(dice_class_1)
        metrics.append(dice_class_2)

    return metrics


def make_custom_loss(loss):
    if loss == "weighted_sum_loss":
        loss = weighted_sum_loss()
    elif loss == "jaccard_distance":
        loss = jaccard_distance
    elif loss == "weighted_dice_loss":
        loss = weighted_dice_coefficient_loss
    elif loss == "weighted_categorical_crossentropy":
        loss = weighted_categorical_crossentropy()

    return loss


def get_optimizer(clipnorm, clipvalue, lr):
    if clipnorm is None and clipvalue is None:
        return Adam(lr=lr)
    elif clipnorm is None:
        return Adam(lr=lr, clipvalue=clipvalue)
    else:
        return Adam(lr=lr, clipnorm=clipnorm, clipvalue=clipvalue)


def make_scores(y, y_pred, scores):
    scores_f = [(x, get_score(x)(y, y_pred)) for x in scores]
    return scores_f


def run_single_test(algorithm_def, x_test, y_test,scores, kwargs):
    enc_model = algorithm_def.get_finetuning_model()

    pred_model = apply_prediction_model(input_shape=enc_model.outputs[0].shape[1:], algorithm_instance=algorithm_def,
                                        **kwargs)

    outputs = pred_model(enc_model.outputs)
    model = Model(inputs=enc_model.inputs[0], outputs=outputs)
    model.load_weights(r"D:/users/zeevh/self_supervised/models/all_base_brats/cpc_weights_test_11/weights-improvement-003.hdf5")

    y_pred = model.predict(x_test, batch_size=1)
    scores_f = make_scores(y_test, y_pred, scores)
    with open(r"D:/users/zeevh/self_supervised/models/all_base_brats/cpc_weights_test_final/test.npy", 'wb') as f:
        np.save(f,y_pred)

def write_result(base_path, row):
    with open(base_path / "results.csv", "a") as csvfile:
        result_writer = csv.writer(csvfile, delimiter=",")
        result_writer.writerow(row)


class MaxTriesExceeded(Exception):
    def __init__(self, func, *args):
        self.func = func
        if args:
            self.max_tries = args[0]

    def __str__(self):
        return f'Maximum amount of tries ({self.max_tries}) exceeded for {self.func}.'


def try_until_no_nan(func, max_tries=4):
    for _ in range(max_tries):
        try:
            return func()
        except NaNLossError:
            print(f"Encountered NaN-Loss in {func}")
    raise MaxTriesExceeded(func, max_tries)


def run_complex_test(
        algorithm,
        dataset_name,
        root_config_file,
        model_checkpoint,
        epochs_initialized=5,
        epochs_random=5,
        epochs_frozen=5,
        repetitions=2,
        batch_size=8,
        exp_splits=(100, 10, 1),
        lr=1e-3,
        epochs_warmup=2,
        scores=("qw_kappa",),
        loss="mse",
        metrics=("mse",),
        clipnorm=None,
        clipvalue=None,
        do_cross_val=False,
        **kwargs,
):
    model_checkpoint = expanduser(model_checkpoint)
    if os.path.isdir(model_checkpoint):
        weight_files = list(Path(model_checkpoint).glob("weights-improvement*.hdf5"))

        if epochs_initialized > 0 or epochs_frozen > 0:
            assert len(weight_files) > 0, "empty directory!"

        weight_files.sort()
        model_checkpoint = str(weight_files[-1])

    kwargs["model_checkpoint"] = model_checkpoint
    kwargs["root_config_file"] = root_config_file
    metrics = list(metrics)

    working_dir = get_writing_path(
        Path(model_checkpoint).expanduser().parent
        / (Path(model_checkpoint).expanduser().stem + "_test"),
        root_config_file,
    )

    algorithm_def = keras_algorithm_list[algorithm].create_instance(**kwargs)
    data_loader = StandardDataLoader(dataset_name, batch_size, algorithm_def, **kwargs)


        # for i in range(repetitions):
        #     logging_base_path = working_dir / "logs"
        #
        #     # Use the same seed for all experiments in one repetition
        #     tf.random.set_seed(i)
        #     np.random.seed(i)
        #     random.seed(i)

    gen_train, gen_val, x_test, y_test = data_loader.get_dataset(0, 100*0.01)
    run_single_test(algorithm_def, x_test, y_test,scores, kwargs)


def main():
    init(run_complex_test, "test")


if __name__ == "__main__":
    main()
