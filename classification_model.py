from keras import models, layers, losses


class SklearnClassificationModel:

    def __init__(self, input_layer):
        self.model = models.Sequential()
        self.model.add(input_layer)

    def add_custom_layer(self, layer):
        self.model.add(layer)

    # model building is extremely sensitive to input/output sizes
    def add_type1_block(self, filters=32):
        self.model.add(layers.Conv2D(
            filters, (3, 3), activation='relu', padding='same', name="type1_Conv2D_1"))
        self.model.add(layers.MaxPooling2D(
            (2, 2), name="type1_MaxPooling2D_1"))

    def add_type2_block(self, filters=64):
        self.model.add(layers.Conv2D(
            filters, (3, 3), activation='relu', padding='same', name="type2_Conv2D_1"))
        self.model.add(layers.Conv2D(
            filters, (3, 3), activation='relu', padding='same', name="type2_Conv2D_2"))
        self.model.add(layers.MaxPooling2D(
            (2, 2), name="type2_MaxPooling2D_1"))

    def add_type3_block(self, filters=128):
        self.model.add(layers.Conv2D(
            filters, (3, 3), activation='relu', padding='same', name="type3_Conv2D_1"))
        self.model.add(layers.Conv2D(
            filters, (3, 3), activation='relu', padding='same', name="type3_Conv2D_2"))
        self.model.add(layers.MaxPooling2D(
            (2, 2), name="type3_MaxPooling2D_1"))

    def add_type4_block(self):
        self.model.add(layers.GlobalAveragePooling2D())
        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(10, activation='softmax'))

    def compile_model(self, optimizer, loss, metrics):
        self.model.compile(
            optimizer=optimizer, loss=loss, metrics=metrics
        )
        return self.model
