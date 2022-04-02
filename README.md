# FER_pytorch
In this repo a facial emotion recognition pipeline is constructed and deployed. 
The pipeline is detecting faces using Haar cascade features using OpenCV implementation. Then preprocessing the data to adjusted it to the required format for the model.
The preprocessed image is feeded to a VGG-like conv net implemented using pytorch and has been trained on the dataset FER2013
