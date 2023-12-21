import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def preprocess_image(image_path):
    image = cv2.imread(image_path)

    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    enhanced_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    noise_removed_img = cv2.GaussianBlur(enhanced_img, (5, 5), 0)

    kernel = np.ones((5, 5), np.uint8)
    erosion_img = cv2.erode(noise_removed_img, kernel, iterations=1)
    dilation_img = cv2.dilate(erosion_img, kernel, iterations=1)

    gray_img = cv2.cvtColor(dilation_img, cv2.COLOR_BGR2GRAY)
    _, segmented_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

    return segmented_img


def resize_image(image, target_size):
    h, w = image.shape[:2]
    scale = target_size / max(h, w)

    new_h, new_w = int(h * scale), int(w * scale)
    resized_image = cv2.resize(image, (new_w, new_h))

    delta_w = target_size - new_w
    delta_h = target_size - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return padded_image


target_size = 224


def load_dataset(image_folder, annotation_folder):
    dataset = []

    for image_file in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_file)
        annotation_path = os.path.join(annotation_folder, os.path.splitext(image_file)[0] + '.txt')

        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as file:
                annotations = file.readlines()
                annotations = [line.strip().split() for line in annotations]
                annotations = [(int(cls), float(x1), float(y1), float(x2), float(y2)) for cls, x1, y1, x2, y2 in annotations]
            if annotations:

                preprocessed_image = resize_image(preprocess_image(image_path), target_size)

                dataset.append((preprocessed_image, annotations))

    return dataset

image_folder = './dataset/images'
annotation_folder = './dataset/annotations'
dataset = load_dataset(image_folder, annotation_folder)

images, annotations = zip(*dataset)

images = np.array(images)
annotations = np.array(annotations)

train_images, test_images, train_annotations, test_annotations = \
    train_test_split(images, annotations, test_size=0.3, random_state=42)

model = Sequential([
    Conv2D(32, kernel_size=3, activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=2),
    Conv2D(64, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(np.array(train_images), np.array(train_annotations), validation_data=(np.array(test_images), np.array(test_annotations)), epochs=30)

def plot_model(history):
    train_loss = history.history['loss']
    train_accuracy = history.history['accuracy']
    epochs = range(1, 31)

    plt.figure(figsize=(12, 5))

    # Plot for training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Training Loss', color='blue')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot for training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, label='Training Accuracy', color='blue')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_model(history)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(np.array(train_images), np.array(train_annotations), validation_data=(np.array(test_images), np.array(test_annotations)), epochs=30)
plot_model(history)

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(np.array(train_images), np.array(train_annotations), validation_data=(np.array(test_images), np.array(test_annotations)), epochs=30)
plot_model(history)

base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(np.array(train_images), np.array(train_annotations), validation_data=(np.array(test_images), np.array(test_annotations)), epochs=30)
plot_model(history)

