from sklearn.preprocessing import StandardScaler, MinMaxScaler
from data_generation import mnist
import tensorflow as tf

def standardize(data):
    scaler = StandardScaler()
    scaler.fit(data)
    standardized_data = scaler.transform(data)

    return standardized_data, scaler


def normalize(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    normalized_data = scaler.transform(data)

    return normalized_data, scaler

def standardize_MNIST(data):
    scaler = mnist.MNIST_Scaler()
    standardized_data = scaler.transform(data)

    return standardized_data, scaler

def preprocess(dataset, batch_size, preprocessing):
    dataset, scaler = preprocessing(dataset) # Normalize the images to [-1, 1]

    # Batch and shuffle the data
    BUFFER_SIZE = dataset.shape[0] # Number of elements in dataset

    train_dataset = tf.data.Dataset.from_tensor_slices(dataset).shuffle(BUFFER_SIZE).batch(batch_size, drop_remainder=True)

    return train_dataset, scaler