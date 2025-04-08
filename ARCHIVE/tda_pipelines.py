from sklearn import set_config
from sklearn.pipeline import make_pipeline, make_union, FeatureUnion, Pipeline
from gtda.diagrams import Scaler, PersistenceImage
from gtda.images import HeightFiltration, Binarizer, RadialFiltration
from gtda.images import DensityFiltration, DilationFiltration, ErosionFiltration, SignedDistanceFiltration
from gtda.homology import CubicalPersistence
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt


def weight_func(x):
    result = 1.2 * np.exp(x) + 0.5 * np.log1p(x) + 0.5 * x
    return result


class SplitRGBChannels(BaseEstimator, TransformerMixin):
    """
    Splits RGB image into 3 separate channels for TDA processing.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        R = X[:, :, :, 0]
        G = X[:, :, :, 1]
        B = X[:, :, :, 2]
        return [R, G, B]


class ImageScalerAndFlattener(BaseEstimator, TransformerMixin):
    """
    Helper class for vector stitching pipeline.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        scaled_and_flattened_images = []
        for image in X:
            min_val, max_val = image.min(), image.max()
            scaled_image = (image - min_val) / (max_val -
                                                min_val) if max_val - min_val > 0 else image
            scaled_and_flattened_images.append(scaled_image.flatten())
        return np.array(scaled_and_flattened_images).reshape(-1, 1, 784)


class CombineTDAWithRGBImages(BaseEstimator, TransformerMixin):
    """
    Combines TDA features from all 3 RGB channels and stitches them together with raw images
    along the last dimension (width).
    """

    def __init__(self, tda_pipeline, input_height=32, input_width=32):
        self.tda_pipeline = tda_pipeline
        self.input_height = input_height
        self.input_width = input_width

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # 1. Split RGB channels
        channels = SplitRGBChannels().transform(X)  # Lista trzech (N, 32, 32)

        tda_features_combined = []

        # 2. Przetwarzanie każdego kanału przez TDA pipeline
        for channel in channels:
            # Kanał do postaci zgodnej z pipeline
            channel = channel.reshape(-1, self.input_height, self.input_width)
            tda_features = self.tda_pipeline.fit_transform(channel)
            # (N, 42, 32, 32) / (N, 14, 32, 32)
            tda_features_combined.append(tda_features)

        # 3. Konkatenacja kanałów TDA wzdłuż osi kanałów
        tda_combined = np.concatenate(
            tda_features_combined, axis=1)  # (N, 126, 32, 32) / (N, 42, 32, 32)

        # 4. Surowe obrazy
        # Przekształcenie na (N, 3, 32, 32)
        raw_images = X.transpose(0, 3, 1, 2)
        raw_images_expanded = np.repeat(
            raw_images, repeats=2, axis=1)  # Powielenie do 126 kanałów / 42 kanałów

        # 5. Konkatenacja wzdłuż osi szerokości (dim=3)
        combined_features = np.concatenate(
            (tda_combined, raw_images_expanded), axis=3)  # (N, 42, 32, 64)

        transposed_combined_features = combined_features.transpose(0, 2, 3, 1)

        return transposed_combined_features


def VECTOR_STITCHING_PI_Pipeline_RGB(dir_list=None, cen_list=None, binarizer_threshold=0.4, bins=28, sig=0.15, input_height=28, input_width=28):
    """
    Returns a pipeline that extracts topological features in the form of persistence images from RGB images.
    Combines TDA features from all three channels.
    """
    direction_list = [[1, 0], [0, 1], [-1, 0], [0, -1]]

    # Creating a list of all filtrations
    """
    filtration_list = (
        [HeightFiltration(direction=np.array(direction), n_jobs=-1) for direction in direction_list] +
        [RadialFiltration(center=np.array([16, 16]), n_jobs=-1)] +
        [DensityFiltration(n_jobs=-1), SignedDistanceFiltration(n_jobs=-1)]
    )
    """

    filtration_list = (
        [RadialFiltration(center=np.array([16, 16]), n_jobs=-1)]
    )

    # PD pipeline
    diagram_steps = [
        [Binarizer(threshold=binarizer_threshold, n_jobs=-1),
         filtration, CubicalPersistence(n_jobs=-1), Scaler(n_jobs=-1)]
        for filtration in filtration_list
    ]

    # feature union
    feature_union = make_union(
        PersistenceImage(sigma=sig, n_bins=min(
            input_height, input_width), weight_function=weight_func, n_jobs=-1)
    )

    tda_union = make_union(
        *[make_pipeline(*diagram_step, feature_union)
          for diagram_step in diagram_steps],
        n_jobs=-1
    )

    combined_features = CombineTDAWithRGBImages(
        tda_pipeline=tda_union, input_height=input_height, input_width=input_width
    )

    final_pipeline = Pipeline([('combine_rgb_features', combined_features)])

    return final_pipeline


def display_pipeline(pipeline):
    """
    Function to display the pipeline object.
    """
    set_config(display='diagram')
    print(pipeline)


def transform_data(X_train, X_test_noisy_random, X_test):
    X_train_expanded = np.expand_dims(X_train, -1)
    X_test_noisy_random_expanded = np.expand_dims(X_test_noisy_random, -1)
    X_test_expanded = np.expand_dims(X_test, -1)
    return X_train_expanded, X_test_noisy_random_expanded, X_test_expanded
