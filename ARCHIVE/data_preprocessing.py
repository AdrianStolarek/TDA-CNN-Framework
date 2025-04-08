
#from tda_pipelines import *
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from skimage.util import random_noise
from sklearn.utils import shuffle
import pickle

# load pipelines


def load_cifar10_batch(batch_path):

    with open(batch_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    data = batch[b'data']
    labels = batch[b'labels']
    # Reshape the data to (num_samples, 3, 32, 32) and then to (num_samples, 32, 32, 3) for RGB images
    data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return data, labels
