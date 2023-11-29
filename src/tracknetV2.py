from keras.models import *
from keras.layers import *
from keras.activations import *

def TrackNet(input_heigth=288, input_width=512):
    imgs_input = Input(shape=(9, input_heigth, input_width))

    # layer 1
    x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(imgs_input)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # layer 2
    x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x1 = BatchNormalization()(x)

    # layer 3
    x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first')(x1)

    # layer 4
    x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # layer 5
    x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x2 = BatchNormalization()(x)

    # layer 6
    x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first')(x2)

    # layer 7
    x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # layer 8
    x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # layer 9
    x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x3 = BatchNormalization()(x)

    # layer 10
    x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first')(x3)

    # layer 11
    x = Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # layer 12
    x = Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # layer 13
    x = Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # layer 14
    x = concatenate([UpSampling2D((2, 2), data_format='channels_first')(x), x3], axis=1)

    # layer 15
    x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # layer 16
    x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)    

    # layer 17
    x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # layer 18
    x = concatenate([UpSampling2D((2, 2), data_format='channels_first')(x), x2], axis=1)

    # layer 19
    x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # layer 20
    x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # layer 21
    x = concatenate([UpSampling2D((2, 2), data_format='channels_first')(x), x1], axis=1)

    # layer 22
    x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # layer 23
    x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # layer 24
    x = Conv2D(1, (1, 1), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('sigmoid')(x)

    output_shape = Model(imgs_input, x).output_shape
    
    output_height, output_width = output_shape[2], output_shape[3]

    # Reshape the size to (288, 512)
    output = Reshape((output_height, output_width))(x)

    model = Model(imgs_input, output)
    model.ouputWidth = output_width
    model.ouputHeight = output_height

    model.summary()
    return model