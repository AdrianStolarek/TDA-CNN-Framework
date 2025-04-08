
from keras import models, layers, losses


class Raw_Model():
    """
    CNN Model for raw input data (28x28 pixel images).
    """

    # if path to existing model is provided, load it, else create a new model
    def __init__(self, model_path=None) -> None:
        self.model = models.Sequential()

        if model_path:
            # TODO: check if model to be loaded it valid for this type of input data
            self.model = models.load_model(model_path)
        else:
            self.init_network()

    def init_network(self):  # model for raw data input (28x28 pixel images)
        self.model.add(layers.Conv2D(
            32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        self.model.add(layers.Flatten())

        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dropout(0.3))

        self.model.add(layers.Dense(10, activation='softmax'))

        self.model.compile(optimizer='adam',
                           loss=losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])


class Dummy_Model():
    """
    Dummy fully dense model for raw input data (28x28 pixel images).
    """

    # if path to existing model is provided, load it, else create a new model
    def __init__(self, model_path=None) -> None:
        self.model = models.Sequential()

        if model_path:
            # TODO: check if model to be loaded it valid for this type of input data
            self.model = models.load_model(model_path)
        else:
            self.init_network()

    def init_network(self):  # model for raw data input (28x28 pixel images)
        self.model.add(layers.Flatten(input_shape=(28, 28)))
        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dense(128, activation='relu'))

        self.model.add(layers.Dense(10, activation='softmax'))

        self.model.compile(optimizer='adam',
                           loss=losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])


class MiniVGG_Raw_Model2:
    """
    Reduced MiniVGG model with ~500K parameters for raw data.
    """

    def __init__(self, input_shape, model_path=None):
        self.model = models.Sequential()

        if model_path:
            self.model = models.load_model(model_path)
        else:
            self.init_network(input_shape)

    def init_network(self, input_shape):
        # Block 1
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu',
                       padding='same', input_shape=input_shape))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        self.model.add(layers.Flatten())

        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(10, activation='softmax'))

        self.model.compile(
            optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']
        )


class MiniVGG_Raw_Model:
    """
    Reduced MiniVGG model with ~500K parameters for raw data.
    """

    def __init__(self, input_shape, model_path=None,
                 block_1_size=32, block_2_size=64, block_3_size=128, optimizer="adam"):
        self.model = models.Sequential()

        if model_path:
            self.model = models.load_model(model_path)
        else:
            self.init_network(input_shape, block_1_size=32,
                              block_2_size=64, block_3_size=128, optimizer="adam")

    def init_network(self, input_shape, block_1_size, block_2_size, block_3_size, optimizer):
        # Block 1
        self.model.add(layers.Conv2D(32, (7, 7), activation='relu',
                       padding='same', input_shape=input_shape))
        self.model.add(layers.Conv2D(
            480, (3, 3), activation='relu', padding='same'))
        self.model.add(layers.MaxPooling2D((2, 2)))

        # Block 2
        self.model.add(layers.Conv2D(
            192, (3, 3), activation='relu', padding='same'))
        self.model.add(layers.MaxPooling2D((2, 2)))

        # Block 3
        self.model.add(layers.Conv2D(
            160, (3, 3), activation='relu', padding='same'))
        self.model.add(layers.MaxPooling2D((2, 2)))

        # Classification block
        self.model.add(layers.GlobalAveragePooling2D())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(100, activation='softmax'))

        self.model.compile(
            optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy']
        )


class MiniVGG_Raw_Model_Lite:
    """
    Unified MiniVGG model with ~6M parameters.
    """

    def __init__(self, input_shape, model_path=None):
        self.model = models.Sequential()

        if model_path:
            self.model = models.load_model(model_path)
        else:
            self.init_network(input_shape)

    def init_network(self, input_shape):
        # Block 1
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu',
                       padding='same', input_shape=input_shape))
        self.model.add(layers.Conv2D(
            64, (3, 3), activation='relu', padding='same'))
        self.model.add(layers.MaxPooling2D((2, 2)))

        # Block 2
        self.model.add(layers.Conv2D(
            128, (3, 3), activation='relu', padding='same'))
        self.model.add(layers.Conv2D(
            128, (3, 3), activation='relu', padding='same'))
        self.model.add(layers.MaxPooling2D((2, 2)))

        # Block 3
        self.model.add(layers.Conv2D(
            256, (3, 3), activation='relu', padding='same'))
        self.model.add(layers.Conv2D(
            256, (3, 3), activation='relu', padding='same'))
        self.model.add(layers.MaxPooling2D((2, 2)))

        # Block 4
        self.model.add(layers.Conv2D(
            384, (3, 3), activation='relu', padding='same'))
        self.model.add(layers.Conv2D(
            384, (3, 3), activation='relu', padding='same'))
        self.model.add(layers.MaxPooling2D((2, 2)))

        # Classification block
        self.model.add(layers.GlobalAveragePooling2D())
        self.model.add(layers.Dense(512, activation='relu'))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(10, activation='softmax'))

        self.model.compile(
            optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']
        )
