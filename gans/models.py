import tensorflow as tf
from tensorflow.keras import layers

def get_models(model, loss_fn, INPUT_SHAPE, NOISE_DIM):

    def dnn_1d(loss_fn, INPUT_SHAPE, NOISE_DIM):

        if loss_fn.lower() == 'gan':
            def generator(NOISE_DIM, INPUT_SHAPE):
                model = tf.keras.Sequential()
                model.add(layers.Input(shape=(NOISE_DIM,)))

                model.add(layers.Dense(512, use_bias=False))
                model.add(layers.BatchNormalization())
                model.add(layers.LeakyReLU())

                model.add(layers.Dense(512, use_bias=False))
                model.add(layers.BatchNormalization())
                model.add(layers.LeakyReLU())
                model.add(layers.Dropout(0.3))

                model.add(layers.Dense(INPUT_SHAPE[0], use_bias=False))

                return model

        elif loss_fn.lower() == 'wgan_gp':
            def generator(NOISE_DIM, INPUT_SHAPE):
                model = tf.keras.Sequential()
                model.add(layers.Input(shape=(NOISE_DIM,)))

                model.add(layers.Dense(512, use_bias=False))
                model.add(layers.LeakyReLU())

                model.add(layers.Dense(512, use_bias=False))
                model.add(layers.LeakyReLU())
                model.add(layers.Dropout(0.3))

                model.add(layers.Dense(INPUT_SHAPE[0], use_bias=False))

                return model

        def discriminator(INPUT_SHAPE):
                model = tf.keras.Sequential()
                model.add(layers.Input(shape=INPUT_SHAPE))

                model.add(layers.Dense(512))
                model.add(layers.LeakyReLU())
                model.add(layers.Dropout(0.3))

                model.add(layers.Dense(512))
                model.add(layers.LeakyReLU())
                model.add(layers.Dropout(0.3))

                model.add(layers.Dense(INPUT_SHAPE[0]))
                model.add(layers.Dense(1))

                return model

        return generator(NOISE_DIM, INPUT_SHAPE), discriminator(INPUT_SHAPE)

    def dnn_2d(loss_fn, INPUT_SHAPE, NOISE_DIM):

        if loss_fn.lower() == 'gan':
            def generator(NOISE_DIM, INPUT_SHAPE):
                pass

        elif loss_fn.lower() == 'wgan_gp':
            def generator(NOISE_DIM, INPUT_SHAPE):
                pass

        def discriminator(INPUT_SHAPE):
            pass
            
        return generator(NOISE_DIM, INPUT_SHAPE), discriminator(INPUT_SHAPE)
    
    def dcgan_1d(loss_fn, INPUT_SHAPE, NOISE_DIM):

        def generator(NOISE_DIM, INPUT_SHAPE):
            pass
            
        def discriminator(INPUT_SHAPE):
            pass

        return generator(NOISE_DIM, INPUT_SHAPE), discriminator(INPUT_SHAPE)
    
    def dcgan_2d(loss_fn, INPUT_SHAPE, NOISE_DIM):

        if loss_fn.lower() == 'gan':
            def generator(NOISE_DIM, INPUT_SHAPE):
                model = tf.keras.Sequential()
                model.add(layers.Input(shape=(NOISE_DIM,)))

                model.add(layers.Dense(7*7*256, use_bias=False))
                model.add(layers.BatchNormalization())
                model.add(layers.LeakyReLU())

                model.add(layers.Reshape((7, 7, 256)))
                assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

                model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
                assert model.output_shape == (None, 7, 7, 128)
                model.add(layers.BatchNormalization())
                model.add(layers.LeakyReLU())

                model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
                assert model.output_shape == (None, 14, 14, 64)
                model.add(layers.BatchNormalization())
                model.add(layers.LeakyReLU())

                model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
                assert model.output_shape == (None, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2])

                return model

        elif loss_fn.lower() == 'wgan_gp':
            def generator(NOISE_DIM, INPUT_SHAPE):
                model = tf.keras.Sequential()
                model.add(layers.Input(shape=(NOISE_DIM,)))

                model.add(layers.Dense(7*7*256, use_bias=False))
                model.add(layers.LeakyReLU())

                model.add(layers.Reshape((7, 7, 256)))

                model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
                model.add(layers.LeakyReLU())

                model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
                model.add(layers.LeakyReLU())

                model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
                assert model.output_shape == (None, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2])

                return model
            
        def discriminator(INPUT_SHAPE):
            model = tf.keras.Sequential()
            model.add(layers.Input(shape=INPUT_SHAPE))

            model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
            model.add(layers.LeakyReLU())
            model.add(layers.Dropout(0.3))

            model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
            model.add(layers.LeakyReLU())
            model.add(layers.Dropout(0.3))

            model.add(layers.Flatten())
            model.add(layers.Dense(1))

            return model

        return generator(NOISE_DIM, INPUT_SHAPE), discriminator(INPUT_SHAPE)

    if model.lower() == 'dnn_1d':
        generator, discriminator = dnn_1d(loss_fn, INPUT_SHAPE, NOISE_DIM)
    elif model.lower() == 'dnn_2':
        generator, discriminator = dnn_2d(loss_fn, INPUT_SHAPE, NOISE_DIM)
    elif model.lower() == 'dcgan_1d':
        generator, discriminator = dcgan_1d(loss_fn, INPUT_SHAPE, NOISE_DIM)
    elif model.lower() == 'dcgan_2d':
        generator, discriminator = dcgan_2d(loss_fn, INPUT_SHAPE, NOISE_DIM)

    return generator, discriminator
