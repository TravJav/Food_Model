
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

IMAGES = '../Data/food-101/images/'
CLASSES = '../Data/food-101/meta/classes.txt'
TEST = '../Data/food-101/meta/test.txt'
TRAIN = '../Data/food-101/meta/train.txt'


class Train:

    def __init__(self):
        self.images = IMAGES
        self.classes = CLASSES
        self.training_set = TRAIN
        self.test_set = TEST
        self.catagories = CLASSES
        self.img_width = 256
        self.img_height = 256

    def build_model(self):
        print("preparing to load images and the assigned classes...")
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            vertical_flip=False,  # randomly flip images
            horizontal_flip=True,
            validation_split=0.3)

        train_generator = train_datagen.flow_from_directory(
            self.images,
            target_size=(self.img_height, self.img_width),
            color_mode='rgb',
            batch_size=32,
            class_mode='binary',
            subset='training')  # set as training data

        validation_generator = train_datagen.flow_from_directory(
            self.images,  # same directory as training data
            target_size=(self.img_height, self.img_width),
            color_mode='rgb',
            batch_size=32,
            class_mode='binary',
            subset='validation')  # set as validation data

        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=(256, 256, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(101, activation='softmax'))

        filepath = 'model_sentiment.hdf5'
        checkpoint = ModelCheckpoint(filepath,
                                     monitor='val_acc',
                                     verbose=2,
                                     save_best_only=True,
                                     mode='max')

        callbacks_list = [checkpoint]

        model.summary()

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.samples // 32,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples//32,
            callbacks=callbacks_list,
            use_multiprocessing=False,
            verbose=1,
            shuffle=False,
            max_queue_size=10,
            workers=32,
            epochs=5,
            initial_epoch=0)

        model.summary()


train = Train()
train.build_model()