import tensorflow as tf


class DataHandler:

    def __init__(self, dataset_dir,
                 split_dataset=0.2,
                 batch_size=16,
                 target_shape=(128, 128)):
        self.__target_shape = target_shape
        self.__dataset_dir = dataset_dir

        self.__dataset_partitions = split_dataset

        self.__train, self.__validation = self.__generate_dataset(batch_size=batch_size)

    def __generate_dataset(self, batch_size):
        data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=40,
                                                                         width_shift_range=0.2,
                                                                         height_shift_range=0.2,
                                                                         shear_range=0.2,
                                                                         zoom_range=0.2,
                                                                         horizontal_flip=True,
                                                                         fill_mode='nearest',
                                                                         validation_split=self.__dataset_partitions)

        train = data_generator.flow_from_directory(self.__dataset_dir, class_mode="categorical",
                                                   target_size=self.__target_shape, color_mode='rgb',
                                                   batch_size=batch_size, shuffle=True, subset='training')

        validation = data_generator.flow_from_directory(self.__dataset_dir, class_mode="categorical",
                                                        target_size=self.__target_shape, color_mode='rgb',
                                                        batch_size=batch_size, subset='validation')

        return train, validation

    def get_train(self):
        return self.__train

    def get_validation(self):
        return self.__validation

    def get_shape(self):
        return self.__target_shape
