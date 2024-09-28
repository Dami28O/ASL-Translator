import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import random

from sklearn.model_selection import train_test_split

import mediapipe as mp


EPOCHS = 20
IMG_WIDTH = 256
IMG_HEIGHT = 256
NUM_CATEGORIES = 28 # (26 Letters, space and del, not training for nothing despite being in original dataset) 
TEST_SIZE = 0.5


def main():

    # initialising the detector class
    mp_hands = mp.solutions.hands

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python3 SLmodel.py data_directory [model.h5]")

    hands = setupMarker(mp_hands)
    
    # get the tuples
    landmarks, labels = load_data(sys.argv[1], hands)

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels, num_classes=NUM_CATEGORIES)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(landmarks), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        filename = filename + ".h5"
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir, hands):
    """
    Loads image data from the directory 'data dir'

    Iterates over each image passing it into the media pipe landmarker model
    returning the landmarks

    add the landmarks to a list of landmarks 

    return a tuple of (landmarks, labels)
    """

    img_landmarks = []
    labels = []

    for category in os.listdir(data_dir):
        # get to the actual data category folders based on how the data is structred
        category_path = os.path.join(data_dir, category)

        # counter to limit how much data we training on
        counter = 0
        images = []

        # check that the current item is a directory
        if os.path.isdir(category_path):
            
            # generate of every image in the directory
            list_dir = os.listdir(category_path)

            # pick 500 images at random
            while counter < 500:
                image = random.choice(list_dir)

                # if we have already found added that image to the training data group skip that image
                if image in images:
                    continue
                # if we have already got all the images in the directory then end loop
                elif len(images) == len(list_dir):
                    break
                else:
                    # add to list of seen elements
                    images.append(image)

                    # pass into landmarker
                    image_path = os.path.join(category_path, image)
                    landmarks, exists = landmark_hands(image_path, hands)

                     # if the image exists append it to the dataset
                    if exists:
                        # append the data and labels 
                        img_landmarks.append(landmarks)

                        label = labelEncode(category)
                        labels.append(label)

                        counter += 1

            """
            # to overcome  issues with handling too much data
            
            # go through every image
            for image in os.listdir(category_path):
                image_path = os.path.join(category_path, image)

                landmarks, exists = landmark_hands(image_path, hands)
                
                # if the image exists append it to the dataset
                if exists:
                    # append the data and labels 
                    img_landmarks.append(landmarks)

                    label = labelEncode(category)
                    labels.append(label)
            """
    
    return (img_landmarks, labels)


def landmark_hands(img_path, hands):
    """
    resize and normalise images as needed by changing them to numpy arrays
    Get the landmarks for each image and return the list of landmarks 
    """
    # list of the landmarks
    landmarks = []
    flattened_landmarks = []


    # read the image as a numpy array
    image = cv2.imread(img_path)

    desired_size = (IMG_WIDTH, IMG_HEIGHT)

    # a variable to keep track of whether this image actually exists, for the labelling
    exists = False

    # check if the image exists
    if image is not None:
        
        # resize if needed
        current_dim = (image.shape[0], image.shape[1])
        if current_dim != desired_size:
            image = cv2.resize(image, desired_size)

        # convert to image to RGB from BGR
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # perform the detection
        detection_result = hands.process(image_rgb)

        # if we detect something
        if detection_result.multi_hand_landmarks:
            # draw the image
            # annotated_image = image.copy()

            # to avoid images that the model cannot recognise (Case w/ 'B' images)
            exists = True

            # go over every single landmark in in the hand and save to the landmarks list (loop for if we were doing two hands)
            for hand_landmarks in detection_result.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    # access individual x, y, z coordinates
                    landmarks.append([landmark.x, landmark.y, landmark.z])

            # flatten into a 1D array for the NN
            flattened_landmarks = [item for sublist in landmarks for item in sublist]

    return flattened_landmarks, exists

def get_model():
    """
    Returns a compiled MLP model. 
    Where the input layer should be the 1D landmark positions (21 landmarks with x, y, z coordinates = 63)
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    input_shape = (63,)

     # Create the MLP
    model = tf.keras.models.Sequential([

        # Input layer
        tf.keras.layers.InputLayer(input_shape=input_shape, activation='relu'),

        # hidden layer1
        tf.keras.layers.Dense(128, activation='relu'),

        # hidden layer2
        tf.keras.layers.Dense(64, activation='relu'),

        # Add a hidden layer with dropout
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.4),

        # Add an output layer with output units for all the categories
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # Train neural network
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def labelEncode(category):
    """
    a small variable to encode the categories for indexing into the NN
    """
    if category == 'space':
        return 26
    elif category == 'del':
        return 27
    else:
        # get ascii value and rest to have 0-indexing of categories
        ascii = ord(category)
        encoded = ascii - 65

    return encoded
    
def setupMarker(mp_hands):

    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1
    )

    # return the detector to be used by the landmark_hands function
    return hands

if __name__ == "__main__":
    main()