# FER_pytorch
In this repo a facial emotion recognition pipeline is constructed and deployed.\

The pipeline is detecting faces using Haar cascade features using OpenCV implementation. Then preprocessing the data to adjusted it to the required format for the model.\
The preprocessed image is feeded to a VGG-like convnet implemented using pytorch and has been trained on the dataset [FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) with 67.5% accuracy on test set.\
