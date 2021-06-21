import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from random import choice, sample
from tqdm import tqdm


class DatasetHandler:

    def __init__(self, train_path: str, test_path: str,
                 train_part: float = 1., test_part: float = 1.,
                 batch_size=64,
                 target_shape=(400, 400)):
        self.__target_shape = target_shape

        full_train_data = pd.read_csv(train_path, sep=' ')
        full_test_data = pd.read_csv(test_path, sep=' ')

        part_train_data_indexes = list(full_train_data.index)
        part_test_data_indexes = list(full_test_data.index)

        len_train_part = int(train_part * len(part_train_data_indexes))
        len_test_part = int(test_part * len(part_test_data_indexes))

        train_source = full_train_data.loc[sample(part_train_data_indexes, len_train_part)]
        test_source = full_test_data.loc[sample(part_test_data_indexes, len_test_part)]

        del full_train_data, full_test_data

        # Train/test triplets

        tqdm.write(f'Train generating')
        train_triplets = self.__generate_triplets(train_source)
        self.__train_dataset = self.__seal_dataset(train_triplets)
        self.__train_dataset = self.__train_dataset.batch(batch_size).prefetch(2)
        tqdm.write(f'Test generating')
        test_triplets = self.__generate_triplets(test_source)
        self.__test_dataset = self.__seal_dataset(test_triplets)
        self.__test_dataset = self.__test_dataset.batch(batch_size).prefetch(2)


    def __form_triplet(self, ind: int, data: pd.DataFrame):
        anchor = data.iloc[ind]
        similar_indexes = data.loc[(data.class_id == anchor.class_id) & (data.image_id != anchor.image_id)].index
        if len(similar_indexes) == 0:
            similar_indexes = data.loc[(data.super_class_id == anchor.super_class_id) & (data.image_id != anchor.image_id)].index
        positive = data.loc[choice(similar_indexes)]
        different_indexes = data.drop(index=data.loc[data.class_id == anchor.class_id].index).index
        negative = data.loc[choice(different_indexes)]

        return anchor, positive, negative


    def __generate_triplets(self, data: pd.DataFrame):
        triplets = {'anchors': [], 'positive': [], 'negative': []}
        for i in tqdm(range(data.shape[0])):
            anchor, positive, negative = self.__form_triplet(i, data)
            triplets['anchors'].append(f'{DATASET_PATH}{anchor["path"]}')
            triplets['positive'].append(f'{DATASET_PATH}{positive["path"]}')
            triplets['negative'].append(f'{DATASET_PATH}{negative["path"]}')
        return triplets


    def __seal_dataset(self, data: dict):
        anchor_dataset = tf.data.Dataset.from_tensor_slices(data['anchors'])
        positive_dataset = tf.data.Dataset.from_tensor_slices(data['positive'])
        negative_dataset = tf.data.Dataset.from_tensor_slices(data['negative'])

        triplets_path_dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
        triplets_images_dataset = triplets_path_dataset.map(self.__preprocess_triplets)

        return triplets_images_dataset


    def __preprocess_image(sekf, filename: tf.Tensor):
        """
        Load the specified file as a JPEG image, preprocess it and
        resize it to the target shape.
            """


        image_string = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, (400, 400))

        return image


    @tf.autograph.experimental.do_not_convert
    def __preprocess_triplets(self, anchor, positive, negative):
        """
        Given the filenames corresponding to the three images, load and
        preprocess them.
        """

        return (
            self.__preprocess_image(anchor),
            self.__preprocess_image(positive),
            self.__preprocess_image(negative),
        )


    def get_target_shape(self):
        return self.__target_shape

    def get_training_data(self):
        return self.__train_dataset

    def get_validation_data(self):
        return self.__test_dataset
