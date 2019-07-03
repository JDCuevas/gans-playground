import os
import tensorflow as tf
import h5py

class MNISTDS:

    def __init__(self):
        pass

    def generate_dataset(self):
        (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()

        # Reshape to (n_samples, width, height, n_channels)
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
        
        if not os.path.exists(os.path.dirname('./datasets/')):
            try:
                os.makedirs(os.path.dirname('./datasets/'))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        with h5py.File('./datasets/mnist.hdf5', 'w') as f:
            f.create_dataset("images", data=train_images)

    def load_dataset(self):
        try:
            with h5py.File('./datasets/mnist.hdf5', 'r') as f:
                train_dataset = f['images'][:]

                
        except OSError:
            self.generate_dataset()

            with h5py.File('./datasets/mnist.hdf5', 'r') as f:
                train_dataset = f['images'][:]

        return train_dataset

class MNIST_Scaler():
    
    def __init__(self):
        pass

    def transform(self, image_data):
        return (image_data - 127.5) / 127.5

    def inverse_transform(self, image_data):
        return (image_data * 127.5) + 127.5