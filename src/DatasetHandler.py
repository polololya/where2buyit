import pandas as pd
import tensorflow as tf
from random import choice, sample, randint
from tqdm import tqdm

FULL_DATASET_FILE = 'Ebay_info.txt'


class DatasetHandler:
    """DataHandler - класс для получения и обработки StandfordDataset для модели основанной на TripletLoss

    Attributes:
    -----------
    dataset_dir : str
            путь к папке Standford_Online_Products (Пример: ../Data/Standford_Online_Products)

    split_dataset : tuple(int, int)
            отношение частей train и test

    batch_size : int
            количество триплетов в батче

    target_shape : tuple(int, int)
            размер в который будут переведены изображения

    """

    def __init__(self, dataset_dir,
                 split_dataset=(0.8, 0.2),
                 dataset_part=1,
                 batch_size=64,
                 target_shape=(400, 400)):

        self.__target_shape = target_shape
        self.dataset_dir = dataset_dir

        full_dataset = pd.read_csv(f'{self.dataset_dir}{FULL_DATASET_FILE}', sep=' ')
        self.__dataset_partitions = split_dataset

        df_train, df_test = self.__split_dataset(full_dataset, dataset_part)

        # Train/test triplets

        tqdm.write(f'Train generating')
        train_triplets = self.__generate_triplets(df_train)
        self.__train_dataset = self.__seal_dataset(train_triplets)
        self.__train_dataset = self.__train_dataset.batch(batch_size).prefetch(2)
        tqdm.write(f'Test generating')
        test_triplets = self.__generate_triplets(df_test)
        self.__test_dataset = self.__seal_dataset(test_triplets)
        self.__test_dataset = self.__test_dataset.batch(batch_size).prefetch(2)

    def __split_dataset(self, data: pd.DataFrame, dataset_part: float):
        """
        Деление всего датасета на train/test в зависимости с dataset_part и split_dataset, переданным в параметры класса
        """
        super_classes = list(data.super_class_id.unique())
        df_train = data.copy()
        df_test = data.copy()
        for super_class in super_classes:
            super_class_indexes = list(data.loc[data.super_class_id == super_class].index)
            dropped_index = sample(super_class_indexes, int(dataset_part * len(super_class_indexes)))
            dropped_index = list(set(super_class_indexes) - set(dropped_index))
            train_super_class_indexes = sample(super_class_indexes,
                                               int(self.__dataset_partitions[0] * len(super_class_indexes)))

            test_super_class_indexes = list(set(super_class_indexes) - set(train_super_class_indexes))

            df_train.drop(index=test_super_class_indexes + dropped_index, inplace=True)
            df_test.drop(index=train_super_class_indexes + dropped_index, inplace=True)
        return df_train, df_test

    def __form_triplet(self, ind: int, data: pd.DataFrame):
        """
        Формирование триплета. Для выбранного изображения берется изображение из его класса, если такое отсутствует, то
        берется из суперласса. Отличное от выбранного изображение берется таким, чтобы оно не было в том же классе, что
        и выбранный.
        """
        anchor = data.iloc[ind]
        similar_indexes = data.loc[(data.class_id == anchor.class_id) & (data.image_id != anchor.image_id)].index
        if len(similar_indexes) == 0:
            similar_indexes = data.loc[(data.super_class_id == anchor.super_class_id)].index
        positive = data.loc[choice(similar_indexes)]
        different_indexes = data.drop(index=data.loc[data.class_id == anchor.class_id].index).index
        negative = data.loc[choice(different_indexes)]

        return anchor, positive, negative

    def __generate_triplets(self, data: pd.DataFrame):
        """
        Генерация триплетов, на данном этапе хранятся лишь пути к изображениям
        """
        triplets = {'anchors': [], 'positive': [], 'negative': []}
        for i in tqdm(range(data.shape[0])):
            anchor, positive, negative = self.__form_triplet(i, data)
            triplets['anchors'].append(f'{self.dataset_dir}{anchor["path"]}')
            triplets['positive'].append(f'{self.dataset_dir}{positive["path"]}')
            triplets['negative'].append(f'{self.dataset_dir}{negative["path"]}')
        return triplets

    def __seal_dataset(self, data: dict):
        """
        Получение триплета из целевого изображения, похожего на него и отличного от него.
        """
        anchor_dataset = tf.data.Dataset.from_tensor_slices(data['anchors'])
        positive_dataset = tf.data.Dataset.from_tensor_slices(data['positive'])
        negative_dataset = tf.data.Dataset.from_tensor_slices(data['negative'])

        triplets_path_dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
        triplets_images_dataset = triplets_path_dataset.map(self.__preprocess_triplets).map(
            self.__augmentation_triplets)

        return triplets_images_dataset

    def __augmentation_triplets(self, anchor, positive, negative):

        """
        Аугментация каждого изображения из триплета
        """

        return (
            self.__augmentation_image(anchor),
            self.__augmentation_image(positive),
            self.__augmentation_image(negative),
        )

    def __augmentation_image(self, image):

        """
        Аугментация изображения

        random_flip_left_right - случайное отражение по оси Y
        random_flip_up_down - случайное отражение по оси X
        random_brightness - случайное изменение яяркости
        random_contrast - случайное изменение контраста
        random_saturation -  случайное изменение насыщенности
        rot90 - переворот на 90 градусов случайное кол-во раз
        """

        aug_image = tf.image.random_flip_left_right(image)
        aug_image = tf.image.random_flip_up_down(aug_image)
        aug_image = tf.image.random_brightness(aug_image, max_delta=0.3)
        aug_image = tf.image.random_contrast(aug_image, lower=0.6, upper=1)
        aug_image = tf.image.random_saturation(aug_image, 0.6, 1)
        aug_image = tf.image.rot90(aug_image, k=randint(0, 3))
        return aug_image

    def __preprocess_image(self, filename: tf.Tensor):
        """
        Загрузка изображения, декодирование, перевод значений в числа с плавающей точкой, а также изменение размера
        """

        image_string = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, self.__target_shape)

        return image

    @tf.autograph.experimental.do_not_convert
    def __preprocess_triplets(self, anchor, positive, negative):
        """
        Метод для обработки каждого изображения из триплета
        """

        return (
            self.__preprocess_image(anchor),
            self.__preprocess_image(positive),
            self.__preprocess_image(negative),
        )

    """
    Метод для поучения установленного размера изображений
    """

    def get_target_shape(self):
        return self.__target_shape

    """
    Метод ждя получения train dataset
    """

    def get_training_data(self):
        return self.__train_dataset

    """
    Метод для получения test dataset
    
    
    """

    def get_validation_data(self):
        return self.__test_dataset
