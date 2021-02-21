import tensorflow as tf

class BasicNetwork(tf.keras.Model):
    """
    A Basic MNIST like network to start your training quickly. With a fresh run on a GPU this model can
    give you around 98.99% testing accuracy on a MNIST Problem. (99.7% is the global highest)
        Args:
            - ``num_classes``: Total number of classes for the output labels
    """

    def __init__(self, num_classes):
        super(BasicNetwork, self).__init__()
        
        self.conv64 = tf.keras.layers.Conv2D(64, kernel_size=2, strides=1)
        self.conv32 = tf.keras.layers.Conv2D(32, kernel_size=2, strides=1)

        self.bn = tf.keras.layers.BatchNormalization()
        self.maxpool = tf.keras.layers.MaxPooling2D((2,2))
        self.flatten = tf.keras.layers.Flatten()

        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')

        self.relu_activation = tf.keras.layers.Activation('relu')

        self.out = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, input_tensor):
        x = self.conv32(input_tensor)
        x = self.bn(x)
        x = self.relu_activation(x)

        x = self.maxpool(x)
        
        x = self.conv32(input_tensor)
        x = self.bn(x)
        x = self.relu_activation(x)

        x = self.maxpool(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        
        return self.out(x)
