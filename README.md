# Facial Expression Recognition Using pytorch
In this repo a facial emotion recognition pipeline is constructed and deployed.\

The pipeline is detecting faces using Haar cascade features using OpenCV implementation. Then preprocessing the data to adjusted it to the required format for the model.\
The preprocessed image is feeded to a VGG-like convnet implemented using pytorch and has been trained on the dataset [FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) with 67.5% accuracy on validation set.

The original dataset and paper were published in 2013 under the name of <b>"Challenges in Representation Learning: A report on three machine learning contests" </b> by I Goodfellow, Y. Bengio et al. - [arXiv 2013](https://arxiv.org/pdf/1307.0414v1.pdf).

This [link](https://drive.google.com/file/d/1G3wtZz1TZ6RpmaGXnflmIfK9H3WHRGHI/view?usp=sharing) provides the serialized model to be downloaded in the main directory in order to run the code.

An explaining video and Testing is available on this [link](https://www.linkedin.com/posts/hussein-barakat-576aa5106_machinelearning-ai-practicemakesperfect-activity-6917894549897900032-3hTp?utm_source=linkedin_share&utm_medium=member_desktop_web)
