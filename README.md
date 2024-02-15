# Deep Fake Detection Project
## Overview
This project aims to develop a deep learning model for detecting deep fake videos, which are synthetic videos generated using deep learning techniques to manipulate or replace the appearance of people in existing videos. The model is trained on a dataset containing both real and fake videos.

## Dataset
Dataset consists of 320 real videos and 320 deep fake videos.
### Links to dataset-
#### Deep Fake Videos - https://zenodo.org/record/4068245/files/DeepfakeTIMIT.tar.gz?download=1
#### Real Videos - https://lp-prod-resources.s3.amazonaws.com/other/detectingdeepfakes/VidTIMIT.zip

## Model Architecture
The model architecture used in this project is based on a combination of convolutional neural networks (CNNs) and recurrent neural networks (RNNs). Specifically, a ResNet50 model pretrained on ImageNet is used as the feature extractor for the video frames. This model is followed by a Convolutional Long Short-Term Memory (ConvLSTM) layer to capture temporal dependencies in the video data. Finally, a dense layer with a sigmoid activation function is added to output the probability of a video being a deep fake.

## Training
The model is trained using a custom training data generator that loads the video frames from the npz files. The training process involves optimizing the binary cross-entropy loss function using the Adam optimizer.

## Dependencies
Python 3.7+,
TensorFlow,
CV2,
NumPy,
OpenCV,
Matplotlib,
Time,
Shutil,
Concurrent,
OS,
Pathlib,
MTCNN

## Usage
Clone the repository.
Download the dataset and preprocess the videos (if necessary).
Train the model using the provided training script.
Evaluate the model's performance on the validation dataset.
Use the trained model for inference on new video data.

## Contributors
Noel William
