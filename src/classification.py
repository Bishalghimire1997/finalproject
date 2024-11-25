import tensorflow as tf
from PIL import Image,ImageDraw, ImageFont
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import os
import numpy as np
class Classification():
    def __init__(self):
          self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    def get_mnist_fashion_data(self):
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (train_images, train_labels),(test_images, test_labels) = fashion_mnist.load_data()
        return (train_images, train_labels),(test_images, test_labels)
    def convert_to_mnist_format(self, image_path):
        """Converts a single image to the Fashion MNIST format.

        Args:
            image_path: Path to the image file.
            label: The corresponding label for the image.

        Returns:
            A tuple containing the image as a NumPy array and its label.
            Returns None if the image cannot be processed.
        """
        try:
            img = Image.open(image_path).convert('L')  # Convert to grayscale
            img = img.resize((28, 28))  # Resize to 28x28 pixels
            img_array = np.array(img)
            img= 255-img_array
            img_array = np.array(img)
            return img_array
        except (FileNotFoundError, OSError, ValueError):
            print(f"Error processing image: {image_path}")
            return None
    def data_visualization(self):
       
        (train_images, train_labels),(test_images, test_labels) = self.get_mnist_fashion_data()
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(train_images[i], cmap=plt.cm.binary)
            plt.xlabel(self.class_names[train_labels[i]])
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
        plt.xticks(unique, self.class_names, rotation=45, ha='right') # Set x-axis labels to class names
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
        layers=tf.keras.layers
        model = models.Sequential()


        model.add(layers.InputLayer(input_shape=(28*28)))
        model.add(layers.Dense(500, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(250, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(100, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(50, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(10, activation='softmax'))
        model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        history = model.fit(train_images, train_labels,
                            epochs=60,
                            batch_size=4800,
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
        return model
    def predict_image(self,model, image):
        if image.shape != (28, 28):
            raise ValueError("Input image must have shape (28, 28)")
        image_flattened = image.reshape(1, 28 * 28).astype('float32') / 255
        probabilities = model.predict(image_flattened)
        predicted_class = np.argmax(probabilities)

        return predicted_class, probabilities[0]
    def save_model(self,model):
        model.save('model.h5')
    def load_model(self):
        return tf.keras.models.load_model('model.h5')
    
    
    def preprocessing(self):

        (train_images, train_labels), (test_images, test_labels) = self.get_mnist_fashion_data()
        train_images = train_images.reshape((train_images.shape[0], 28 * 28)).astype('float32')/255
        test_images = test_images.reshape((test_images.shape[0], 28 * 28)).astype('float32')/255
        train_labels = tf.keras.utils.to_categorical(train_labels, 10)
        test_labels = tf.keras.utils.to_categorical(test_labels, 10)
        train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)
        return  [train_images, train_labels, test_images,test_labels,val_images,val_labels]
    
    def plot_confusion_matrix(self, model, test_images, test_labels):
        # Predict the labels for the test set
        predictions = model.predict(test_images)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(test_labels, axis=1)

        # Compute the confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)

        # Plot the confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=obj.class_names, yticklabels=obj.class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()
    def read_image_from_foder_in_mnist_format(self,path):
        labels=[]
        images=[]
        files_with_extensions = []
        try:
            for entry in os.listdir(path):
                entry_path = os.path.join(path, entry)
                if os.path.isfile(entry_path):
                    files_with_extensions.append(entry)
        except FileNotFoundError:
            print(f"Error: Directory '{path}' not found.")
        except PermissionError:
            print(f"Error: Permission denied for directory '{path}'.")
        for i in files_with_extensions:
            images.append(self.convert_to_mnist_format(path+"/"+i))
            labels.append(path)
        return images,labels
    def __add_tag_to_image(self,input_path, output_path, tag_text):
        """
        Adds a tag to an image and saves the tagged image.

        :param input_path: Path to the input image.
        :param output_path: Path to save the tagged image.
        :param tag_text: The text to use as the tag.
        """
        try:
            # Open the image
            img = Image.open(input_path).convert("RGBA")

            # Create a transparent overlay
            txt_overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))

            # Draw the tag on the overlay
            draw = ImageDraw.Draw(txt_overlay)
            font_size = int(img.size[1] * 0.05)  # Font size relative to image height
            font = ImageFont.load_default()
            # Position for the tag (bottom-right corner)
            text_width, text_height = (100,30)
            position = (img.size[0] - text_width - 10, img.size[1] - text_height - 10)

            # Add text with a semi-transparent background
            draw.rectangle(
                [position, (position[0] + text_width, position[1] + text_height)],
                fill=(0, 0, 0, 128),
            )
            draw.text(position, tag_text, font=font, fill=(255, 255, 255, 255))

            # Merge the overlay with the image
            watermarked_image = Image.alpha_composite(img, txt_overlay)

            # Save the tagged image
            watermarked_image.convert("RGB").save(output_path)
            print(f"Image saved with tag at: {output_path}")

        except FileNotFoundError:
            print(f"Error: File not found at {input_path}.")
        except Exception as e:
            print(f"An error occurred: {e}")
    def group(self,path,tag):
        files_with_extensions=[]
        try:
            for entry in os.listdir(path):
                entry_path = os.path.join(path, entry)
                if os.path.isfile(entry_path):
                    files_with_extensions.append(entry)
        except FileNotFoundError:
            print(f"Error: Directory '{path}' not found.")
        except PermissionError:
            print(f"Error: Permission denied for directory '{path}'.")
        for i in range (len(files_with_extensions)):
            print(path+"/"+files_with_extensions[i])
            self.__add_tag_to_image(path+"/"+files_with_extensions[i],"grouped_images/"+files_with_extensions[i],tag[i])



        




    


obj = Classification()
#obj.data_visualization()
preprocesse_data_list = obj.preprocessing()
#model = obj.train_model(preprocesse_data_list)
#obj.save_model(model)
path ="dress"
model = obj.load_model()
images,labels= obj.read_image_from_foder_in_mnist_format(path)
tags=[]
for i in images:
    val, _ = obj.predict_image(model,i)
    tags.append(obj.class_names[val])
obj.group(path,tags)