from keras.models import Model
from keras.layers import Input, Conv2D, Activation, BatchNormalization, MaxPooling2D, UpSampling2D, Reshape, Permute

def TrackNet(input_shape, n_classes=256, input_heigth=360, input_width=640):
    input = Input(shape=(input_shape, input_heigth, input_width))

    # layer 1
    x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(input)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # layer 2
    x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # layer 3
    x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first')(x)

    # layer 4
    x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # layer 5
    x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # layer 6
    x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first')(x)

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
    x = BatchNormalization()(x)

    # layer 10
    x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first')(x)

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
    x = UpSampling2D((2, 2), data_format='channels_first')(x)

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
    x = UpSampling2D((2, 2), data_format='channels_first')(x)

    # layer 19
    x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # layer 20
    x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # layer 21
    x = UpSampling2D((2, 2), data_format='channels_first')(x)

    # layer 22
    x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # layer 23
    x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # layer 24
    x = Conv2D(n_classes, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    output_shape = Model(input, x).output_shape
    output_height, output_width = output_shape[2], output_shape[3]
    print(f"layer 24's output shape: {output_shape[1]} X {output_height} X {output_width}")

    # Reshape the size to (256, 360 * 640)
    x = Reshape((-1, output_height * output_width))(x)

    # Change dimension order to (360 * 640, 256)
    x = Permute((2, 1))(x)

    output = Activation('softmax')(x)
    
    model = Model(input, output)
    model.ouputWidth = output_width
    model.ouputHeight = output_height

    model.summary()
    return model