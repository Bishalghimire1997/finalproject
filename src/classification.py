import tensorflow as tf
from matplotlib import pyplot as plt
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
       # Display the first 25 images from the training set
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(train_images[i], cmap=plt.cm.binary)
            plt.xlabel(class_names[train_labels[i]])
        plt.show()

        # Show image shape and pixel values
        print("Shape of the training image:", train_images.shape)
        print("Pixel values range from", np.min(train_images), "to", np.max(train_images))


        # Display a histogram of pixel values
        plt.figure()
        plt.hist(train_images.flatten(), bins=50)
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.title('Histogram of Pixel Values')
        plt.show()

        # Display a bar chart of class distribution
        unique, counts = np.unique(train_labels, return_counts=True)
        plt.figure()
        plt.bar(unique, counts)
        plt.xlabel('Class Labels')
        plt.ylabel('Frequency')
        plt.title('Class Distribution in the Training Set')
        plt.xticks(unique, class_names, rotation=45, ha='right') # Set x-axis labels to class names
        plt.tight_layout() # Adjust layout to prevent labels from overlapping
        plt.show()


    def preprocess(self):
        pass

obj = Classification()
obj.data_visualization()