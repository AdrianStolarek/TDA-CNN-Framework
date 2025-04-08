from sklearn.pipeline import make_pipeline, make_union, Pipeline
from gtda.diagrams import Scaler, PersistenceImage
from gtda.images import Binarizer, RadialFiltration
from gtda.homology import CubicalPersistence
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler


def weight_func(x):
    """Funkcja wagowa dla obrazów persystencji."""
    return 1.2 * np.exp(x) + 0.5 * np.log1p(x) + 0.5 * x


class CombineTDAWithGrayImages(BaseEstimator, TransformerMixin):
    """
    Kombinuje cechy TDA z obrazami grayscale dla MNIST.
    """

    def __init__(self, tda_pipeline, input_height=28, input_width=28, w0=0.1, w1=0.9):
        self.tda_pipeline = tda_pipeline
        self.input_height = input_height
        self.input_width = input_width
        self.w0 = w0  # Waga dla PI_H0
        self.w1 = w1  # Waga dla PI_H1

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # 1. Upewniamy się, że X jest w kształcie (N, 28, 28)
        X = X.reshape(-1, self.input_height, self.input_width)

        # 2. Przetwarzanie przez pipeline TDA
        tda_features = self.tda_pipeline.fit_transform(
            X)  # (N, 2, 28, 28) -> H0 i H1

        # 3. Sumowanie ważone H0 i H1
        PI_H0 = tda_features[:, 0, :, :]  # Pierwszy obraz PI_H0
        PI_H1 = tda_features[:, 1, :, :]  # Drugi obraz PI_H1
        weighted_sum = self.w0 * PI_H0 + self.w1 * PI_H1  # Sumowanie ważone

        weighted_sum /= 255.0

        # 4. Surowe obrazy
        raw_images = X[:, :, :, np.newaxis]  # (N, 28, 28, 1)
        tda_layer = weighted_sum[:, :, :, np.newaxis]  # (N, 28, 28, 1)

        # 5. Łączymy dane (N, 28, 28, 2) - surowe obrazy + cechy topologiczne
        combined_features = np.concatenate((raw_images, tda_layer), axis=-1)

        return combined_features


def VECTOR_STITCHING_PI_Pipeline_Gray(binarizer_threshold=0.4, sig=0.25, input_height=28, input_width=28):
    """
    Pipeline do ekstrakcji cech topologicznych z obrazów grayscale (MNIST) z jedną filtracją.
    """

    # Używamy tylko jednej filtracji: RadialFiltration w punkcie [14, 14]
    filtration_list = [RadialFiltration(center=np.array([14, 14]), n_jobs=-1)]

    diagram_steps = [
        [Binarizer(threshold=binarizer_threshold, n_jobs=-1),
         filtration, CubicalPersistence(n_jobs=-1), Scaler(n_jobs=-1)]
        for filtration in filtration_list
    ]

    feature_union = make_union(
        PersistenceImage(sigma=sig, n_bins=min(input_height, input_width),
                         weight_function=weight_func, n_jobs=-1)
    )

    tda_union = make_union(
        *[make_pipeline(*diagram_step, feature_union)
          for diagram_step in diagram_steps],
        n_jobs=-1
    )

    combined_features = CombineTDAWithGrayImages(
        tda_pipeline=tda_union, input_height=input_height, input_width=input_width
    )

    final_pipeline = Pipeline([('combine_gray_features', combined_features)])

    return final_pipeline
