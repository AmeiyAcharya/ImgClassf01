# Happy or Sad Image Classification using DL

This project focuses on classifying images as either happy or sad using deep learning techniques. The code provided utilizes the TensorFlow library in Python to build and train a convolutional neural network (CNN) model for image classification. 

## Table of Contents
1. [Load Data and Setup](#load-data-and-setup)
2. [Preprocessing](#preprocessing)
3. [Building the DL Model](#building-the-dl-model)
4. [Evaluating Performance](#evaluating-performance)
5. [Saving Model for Future Deployment](#saving-model-for-future-deployment)
## Load Data and Setup <a name="load-data-and-setup"></a>

First, we need to set up the environment and load the necessary libraries. We also need to make sure the GPU memory growth is enabled to avoid out of memory errors. 

We then import the required libraries, such as OpenCV, imghdr, and matplotlib, which are used for image processing and visualization. 

Next, we clone the GitHub repository containing the image data for the project. The data is stored in separate directories for "Happy" and "Sad" images. 

We remove any unsuitable or corrupted images from the dataset by checking their file extensions and removing them if they do not match the accepted image types. 

After preprocessing the data, we check the number of remaining images in each class to ensure that the data has been cleaned properly.

## Preprocessing <a name="preprocessing"></a>

In this section, we preprocess the image data before feeding it into the model. 

We start by loading the data using the `image_dataset_from_directory` function from the `tf.keras.utils` module. This function creates a dataset from a directory, with each subdirectory representing a different class. 

We then batch the dataset, resize the images, and create class labels based on the inferred classes from the dataset. This helps streamline the image classification process and prepares the data for training.

## Building the DL Model <a name="building-the-dl-model"></a>

In this section, we build the deep learning model using the Keras Sequential API. 

We create a sequential model object and add layers to it. The layers include convolutional layers with max pooling, followed by a flattening layer and dense layers. The activation functions used are ReLU for the convolutional layers and sigmoid for the final output layer. 

We compile the model using the Adam optimizer and binary cross-entropy loss, and specify accuracy as the evaluation metric. 

After compiling the model, we can use the `summary` function to view the model architecture and the number of parameters. 

We then train the model on the training dataset for a specified number of epochs, with validation data to monitor the model's performance. We also use the TensorBoard callback to log training metrics for visualization.

## Evaluating Performance <a name="evaluating-performance"></a>

In this section, we evaluate the performance of the trained model. 

We import the required metrics from TensorFlow and initialize precision, recall, and accuracy metrics. 

We iterate through the test dataset and predict the class labels for each batch using the trained model. We update the metrics with the true labels and predicted labels. 

Finally, we print the precision, recall, and accuracy values obtained from the evaluation.

## Saving Model for Future Deployment <a name="saving-model-for-future-deployment"></a>

In the last section, we save the trained model for future deployment. We use the `save` function from the `tensorflow.keras.models` module to save the model as an H5 file. 

We can also load the saved model using the `load_model` function and use it to make predictions on new data.
