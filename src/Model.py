import tensorflow as tf
from tensorflow.keras.applications import resnet


class TripletHardLoss(tf.keras.losses.Loss):

    def __init__(self, margin=0.5, squared=True):
        super().__init__()
        self.__margin = margin
        self.__squared = squared

    @tf.autograph.experimental.do_not_convert
    def call(self, y_true, y_pred):
        labels = tf.convert_to_tensor(y_true, name="labels")
        embeddings = tf.convert_to_tensor(y_pred, name="embeddings")

        convert_to_float32 = (
                embeddings.dtype == tf.dtypes.float16 or embeddings.dtype == tf.dtypes.bfloat16
        )
        precise_embeddings = (
            tf.cast(embeddings, tf.dtypes.float32) if convert_to_float32 else embeddings
        )

        batch_size = tf.shape(precise_embeddings)[0]
        adjacency_matrix = tf.matmul(labels, labels, transpose_b=True)
        adjacency_matrix = tf.cast(adjacency_matrix, tf.bool)
        negative_adjacency = tf.math.logical_not(adjacency_matrix)
        negative_adjacency = tf.cast(negative_adjacency, dtype=tf.dtypes.float32)
        positive_adjacency = tf.cast(adjacency_matrix, dtype=tf.dtypes.float32) - tf.eye(batch_size,
                                                                                         dtype=tf.float32)

        dist = self.__pairwise_distances(precise_embeddings)

        distance_embeddings = tf.reshape(dist, [batch_size, batch_size])

        hard_positives = self.__maximum_dist(distance_embeddings, positive_adjacency)
        hard_negatives = self.__minimum_dist(distance_embeddings, negative_adjacency)

        triplet_loss = tf.maximum(hard_positives - hard_negatives + self.__margin, 0.0)
        triplet_loss = tf.reduce_mean(triplet_loss)
        if convert_to_float32:
            return tf.cast(triplet_loss, embeddings.dtype)
        else:
            return triplet_loss

    def __minimum_dist(self, data, mask):
        axis_maximums = tf.math.reduce_max(data, 1, keepdims=True)

        masked_minimums = (
                tf.math.reduce_min(
                    tf.math.multiply(data - axis_maximums, mask), 1, keepdims=True
                )
                + axis_maximums
        )
        return masked_minimums

    def __maximum_dist(self, data, mask):

        masked_maximums = (
            tf.math.reduce_max(
                tf.math.multiply(data, mask), 1, keepdims=True
            )
        )

        return masked_maximums

    def __pairwise_distances(self, embeddings):
        """
        Рассчитать расстояние между векторами вложения
        Args:
                     embeddings: тензор в форме (batch_size, embed_dim)
        Returns:
                     piarwise_distances: тензор формы (batch_size, batch_size)
        """

        dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

        square_norm = tf.linalg.diag_part(dot_product)

        distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)

        distances = tf.maximum(distances, 0.0)

        if not self.__squared:
            mask = tf.cast(tf.equal(distances, 0.0), tf.float32)
            distances = distances + mask * 1e-16

            distances = tf.sqrt(distances)

            distances = distances * (1.0 - mask)
        return distances


class EmbeddingModel(tf.keras.Model):

    def __init__(self, target_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.__base_cnn = resnet.ResNet50(
            weights="imagenet", input_shape=target_shape + (3,), include_top=False
        )

        trainable = False
        for layer in self.__base_cnn.layers:
            if layer.name == "conv5_block1_out":
                trainable = True
            layer.trainable = trainable

        self.__conv_1 = tf.keras.layers.Conv2D(128, kernel_size=(7, 7), padding='same', activation='relu')
        self.__pooling_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.__conv_2 = tf.keras.layers.Conv2D(256, kernel_size=(5, 5), padding='same', activation='relu')
        self.__pooling_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.__flatten_1 = tf.keras.layers.Flatten()
        self.__dense_1 = tf.keras.layers.Dense(1024, activation="relu")
        self.__batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.__dense_2 = tf.keras.layers.Dense(512, activation="relu")
        self.__batch_norm_2 = tf.keras.layers.BatchNormalization()
        self.__embedding = tf.keras.layers.Dense(256, kernel_regularizer='l2')

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, training=None, mask=None):
        layer = self.__base_cnn(inputs)
        layer = self.__conv_1(layer)
        layer = self.__pooling_1(layer)
        layer = self.__conv_2(layer)
        layer = self.__pooling_2(layer)
        layer = self.__flatten_1(layer)
        # layer = self.__dense_1(layer)
        # layer = self.__batch_norm_1(layer)
        layer = self.__dense_2(layer)
        layer = self.__batch_norm_2(layer)
        layer = self.__embedding(layer)
        return layer

    def get_config(self):
        return {
            '__flatten_1': self.__flatten_1,
            '__dense_2': self.__dense_2,
            '__batch_norm_2': self.__batch_norm_2,
            '__embedding': self.__embedding,
            '__base_cnn': self.__base_cnn,
            '__dense_1': self.__dense_1,
            '__batch_norm_1': self.__batch_norm_1,
            '__conv_1': self.__conv_1,
            '__conv_2': self.__conv_2,
            '__pooling_1': self.__pooling_1,
            '__pooling_2': self.__pooling_2
        }

    def from_config(cls, config, custom_objects=None):
        return cls(**config)
