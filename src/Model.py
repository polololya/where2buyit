import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet


class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super(DistanceLayer,self).__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return ap_distance, an_distance



class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.
    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.
    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.05):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs, **kwargs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]


class EmbeddingModel:
    def __init__(self, target_shape):
        base_cnn = resnet.ResNet50(
            weights="imagenet", input_shape=target_shape + (3,), include_top=False
        )

        flatten = layers.Flatten()(base_cnn.output)
        dense = layers.Dense(1024, activation="tanh")(flatten)
        batch_norm = layers.BatchNormalization()(dense)
        dense = layers.Dense(512, activation="tanh")(batch_norm)
        batch_norm = layers.BatchNormalization()(dense)
        embedding = layers.Dense(256)(batch_norm)
        output_layer = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(embedding)

        self.embedding = Model(base_cnn.input, output_layer, name="Embedding")

        trainable = False
        for layer in base_cnn.layers:
            if layer.name == "conv5_block1_out":
                trainable = True
            layer.trainable = trainable

        anchor_input = layers.Input(name="anchor", shape=target_shape + (3,))
        positive_input = layers.Input(name="positive", shape=target_shape + (3,))
        negative_input = layers.Input(name="negative", shape=target_shape + (3,))

        distances = DistanceLayer()(
            self.embedding(resnet.preprocess_input(anchor_input)),
            self.embedding(resnet.preprocess_input(positive_input)),
            self.embedding(resnet.preprocess_input(negative_input)),
        )

        self.siamese_network = Model(
            inputs=[anchor_input, positive_input, negative_input], outputs=distances
        )

    def get_siamese_network(self):
        return self.siamese_network

    def get_embedding(self):
        return self.embedding

    def train(self, train_data, val_data, epoch_num=10, optimizer='adam'):
        siamese_model = SiameseModel(self.siamese_network)
        siamese_model.compile(optimizer=optimizer)
        siamese_model.fit(train_data, epochs=epoch_num, validation_data=val_data)

    def test(self):
        """todo"""

    def inference(self):
        """todo"""
