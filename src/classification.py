import tensorflow as tf
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import os
import numpy as np
class Classification():
    def __init__(self):
        pass
    def get_mnist_fashion_data(self):
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (train_images, train_labels),(test_images, test_labels) = fashion_mnist.load_data()
        return (train_images, train_labels),(test_images, test_labels)
    def data_visualization(self):
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        (train_images, train_labels),(test_images, test_labels) = self.get_mnist_fashion_data()
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(train_images[i], cmap=plt.cm.binary)
            plt.xlabel(class_names[train_labels[i]])
        plt.show()
        print("Shape of the training image:", train_images.shape)
        print("Pixel values range from", np.min(train_images), "to", np.max(train_images))
        plt.figure()
        plt.hist(train_images.flatten(), bins=50)
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.title('Histogram of Pixel Values')
        plt.show()
        unique, counts = np.unique(train_labels, return_counts=True)
        plt.figure()
        plt.bar(unique, counts)
        plt.xlabel('Class Labels')
        plt.ylabel('Frequency')
        plt.title('Class Distribution in the Training Set')
        plt.xticks(unique, class_names, rotation=45, ha='right') # Set x-axis labels to class names
        plt.tight_layout() # Adjust layout to prevent labels from overlapping
        plt.show()

    def train_model(self,data_list):
        train_images =data_list[0]
        train_labels = data_list[1]
        test_images = data_list[2]
        test_labels = data_list[3]
        val_images = data_list[4] 
        val_labels = data_list[5]
        models= tf.keras.models
        layers=tf.keras,layers
        model = models.Sequential()


        model.add(layers.InputLayer(input_shape=(28*28,)))


        model.add(layers.Dense(128, activation='sigmoid'))
        model.add(layers.Dense(128, activation='sigmoid'))
        model.add(layers.Dense(128, activation='sigmoid'))
        model.add(layers.Dense(128, activation='sigmoid'))
        model.add(layers.Dense(128, activation='sigmoid'))


        model.add(layers.Dense(10, activation='softmax'))


        model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])


        history = model.fit(train_images, train_labels,
                            epochs=11,
                            batch_size=480,
                            validation_data=(val_images, val_labels))


        test_loss, test_acc = model.evaluate(test_images, test_labels)
        print(f"Test accuracy: {test_acc * 100:.2f}%")


        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()
        return models
    def preprocessing(self):
        (train_images, train_labels), (test_images, test_labels) = self.get_mnist_fashion_data()
        train_images = train_images.reshape((train_images.shape[0], 28 * 28)).astype('float32') / 255
        test_images = test_images.reshape((test_images.shape[0], 28 * 28)).astype('float32') / 255
        train_labels = tf.keras.utils.to_categorical(train_labels, 10)
        test_labels = tf.keras.utils.to_categorical(test_labels, 10)
        train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)
        return  [train_images, train_labels, test_images,test_labels,val_images,val_labels]

obj = Classification()
obj.data_visualization()
preprocesse_data_list = obj.preprocessing()
obj.train_model(preprocesse_data_list)