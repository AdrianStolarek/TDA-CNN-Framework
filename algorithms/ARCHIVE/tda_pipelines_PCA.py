from sklearn.decomposition import PCA
from sklearn import set_config
from sklearn.pipeline import make_pipeline, make_union, FeatureUnion, Pipeline
from gtda.diagrams import Scaler, PersistenceImage
from gtda.images import Binarizer, RadialFiltration
from gtda.homology import CubicalPersistence
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler


def weight_func(x):
    """Funkcja wagowa dla obrazów persystencji."""
    return 1.2 * np.exp(x) + 0.5 * np.log1p(x) + 0.5 * x


class SplitRGBChannels(BaseEstimator, TransformerMixin):
    """Dzieli obraz RGB na 3 osobne kanały."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [X[:, :, :, i] for i in range(3)]  # Oddzielne kanały R, G, B


class ReduceTDAChannelsWithPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=1):
        self.n_components = n_components
        self.pca_models = [PCA(n_components=n_components) for _ in range(3)]
        self.scalers = [MinMaxScaler() for _ in range(3)]  # Normalizacja

    def fit(self, X, y=None):
        for i in range(3):  # Dla każdego kanału RGB
            h0_h1 = X[i]
            reshaped_h0_h1 = h0_h1.reshape(
                h0_h1.shape[0], 2, -1)  # (N, 2, 1024)
            flattened_h0_h1 = reshaped_h0_h1.reshape(
                h0_h1.shape[0], -1)  # (N, 2048)

            # Normalizacja danych przed PCA
            flattened_h0_h1 = self.scalers[i].fit_transform(flattened_h0_h1)

            # Dopasowanie PCA
            self.pca_models[i].fit(flattened_h0_h1)

        return self

    def transform(self, X):
        reduced_features = []
        for i in range(3):  # Dla każdego kanału RGB
            h0_h1 = X[i]
            reshaped_h0_h1 = h0_h1.reshape(
                h0_h1.shape[0], 2, -1)  # (N, 2, 1024)
            flattened_h0_h1 = reshaped_h0_h1.reshape(
                h0_h1.shape[0], -1)  # (N, 2048)

            # Normalizacja danych przed PCA
            flattened_h0_h1 = self.scalers[i].transform(flattened_h0_h1)

            # Redukcja PCA
            reduced = self.pca_models[i].transform(flattened_h0_h1)  # (N, 1)

            # Rozciągamy do (N, 32, 32)
            reduced = np.repeat(reduced, 1024).reshape(
                h0_h1.shape[0], 32, 32)  # (N, 32, 32)

            reduced_features.append(reduced)

        return np.stack(reduced_features, axis=-1)  # (N, 32, 32, 3)


class CombineTDAWithRGBImages(BaseEstimator, TransformerMixin):
    """
    Combines TDA features from all 3 RGB channels and stitches them together
    with raw images along the channel dimension.
    """

    def __init__(self, tda_pipeline, input_height=32, input_width=32, w0=0.1, w1=0.9):
        self.tda_pipeline = tda_pipeline
        self.input_height = input_height
        self.input_width = input_width
        self.w0 = w0  # Waga dla PI_H0
        self.w1 = w1  # Waga dla PI_H1

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # 1. Podział RGB
        channels = SplitRGBChannels().transform(X)  # Lista trzech (N, 32, 32)

        tda_features_combined = []

        # 2. Przetwarzanie każdego kanału przez TDA pipeline
        for channel in channels:
            channel = channel.reshape(-1, self.input_height, self.input_width)
            tda_features = self.tda_pipeline.fit_transform(
                channel)  # (N, 2, 32, 32) -> H0 i H1

            # 3. Sumowanie ważone H0 i H1
            PI_H0 = tda_features[:, 0, :, :]  # Pierwszy obraz PI_H0
            PI_H1 = tda_features[:, 1, :, :]  # Drugi obraz PI_H1

            weighted_sum = self.w0 * PI_H0 + self.w1 * PI_H1  # Sumowanie ważone
            tda_features_combined.append(weighted_sum)

        # 4. Stworzenie macierzy TDA (N, 3, 32, 32)
        tda_final = np.stack(tda_features_combined, axis=-1)  # (N, 32, 32, 3)

        # 5. Surowe obrazy
        raw_images = X  # (N, 32, 32, 3)

        # 6. Połączenie surowych i PI wzdłuż osi kanałów (N, 32, 32, 6)
        combined_features = np.concatenate((raw_images, tda_final), axis=-1)

        return combined_features


def VECTOR_STITCHING_PI_Pipeline_RGB(binarizer_threshold=0.35, sig=0.3, input_height=32, input_width=32):
    """
    Pipeline do ekstrakcji cech topologicznych z obrazów RGB.
    """

    filtration_list = [RadialFiltration(center=np.array([16, 16]), n_jobs=-1)]

    diagram_steps = [
        [Binarizer(threshold=binarizer_threshold, n_jobs=-1),
         filtration, CubicalPersistence(n_jobs=-1), Scaler(n_jobs=-1)]
        for filtration in filtration_list
    ]

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
    Funkcja do wizualizacji pipeline'u.
    """
    set_config(display='diagram')
    print(pipeline)
