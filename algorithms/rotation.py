import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Lambda, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import Sequential
from self_supervised_3d_tasks.algorithms.algorithm_base import AlgorithmBuilderBase
from self_supervised_3d_tasks.utils.model_utils import (
    apply_encoder_model,
    apply_encoder_model_3d,
    apply_prediction_model)
from self_supervised_3d_tasks.preprocessing.preprocess_rotation import (
    rotate_batch,
    rotate_batch_3d,
)
from self_supervised_3d_tasks.models.unet3d import conv3d_block

class RotationBuilder(AlgorithmBuilderBase):
    def __init__(
            self,
            data_dim=384,
            number_channels=4,
            lr=1e-4,
            data_is_3D=False,
            top_architecture="big_fully",
            **kwargs
    ):
        super(RotationBuilder, self).__init__(data_dim, number_channels, lr, data_is_3D, **kwargs)

        self.image_size = data_dim
        self.img_shape = (self.image_size, self.image_size, number_channels)
        self.img_shape_3d = (
            self.image_size,
            self.image_size,
            self.image_size,
            number_channels,
        )
        self.top_architecture = top_architecture

    def apply_model(self):
        if self.enc_model is None:
            if self.data_is_3D:
                self.enc_model, self.layer_data = apply_encoder_model_3d(self.img_shape_3d, **self.kwargs)
            else:
                self.enc_model, self.layer_data = apply_encoder_model(self.img_shape, **self.kwargs)

        return self.apply_prediction_model_to_encoder(self.enc_model)

    def apply_prediction_model_to_encoder(self, encoder_model):

        if self.data_is_3D:
            x = Dense(10, activation="softmax")
        else:
            x = Dense(4, activation="softmax")
        units = np.prod(encoder_model.outputs[0].shape[1:])
        input_layer = Input((self.img_shape_3d), name="Input")
        hidden = Lambda(lambda x: x, name="anchor_input")(input_layer)
        hidden = conv3d_block(inputs=hidden, filters=4, use_batch_norm=True, dropout=0.5, strides=(2,2,2))
        output = Dense(10, activation="softmax")(Flatten()(encoder_model(hidden)))
        return Model(inputs=input_layer, outputs=output)

        # if self.data_is_3D:
        #     x = Dense(10, activation="softmax")
        # else:
        #     x = Dense(4, activation="softmax")
        # units = np.prod(encoder_model.outputs[0].shape[1:])
        # sub_model = apply_prediction_model((units,), prediction_architecture=self.top_architecture, include_top=False)
        # return Sequential([encoder_model, Flatten(), sub_model, x])

    def get_training_model(self):
        model = self.apply_model()
        model.compile(
            optimizer=Adam(lr=self.lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def get_training_preprocessing(self):
        def f(x, y):  # not using y here, as it gets generated
            return rotate_batch(x, y)

        def f_3d(x, y):
            return rotate_batch_3d(x, y)

        if self.data_is_3D:
            return f_3d, f_3d
        else:
            return f, f

    def purge(self):
        for i in reversed(range(len(self.cleanup_models))):
            del self.cleanup_models[i]
        del self.cleanup_models
        self.cleanup_models = []

    def get_encoder_from_model(self, model):
        return model.layers[7]

    def set_encoder(self, model, encoder):
        self.enc_model = model.layers[7] = encoder
        return model

def create_instance(*params, **kwargs):
    return RotationBuilder(*params, **kwargs)
