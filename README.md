# blood-cell-type-classification
Classification of blood cell types (8 classes) using pretrained Convolutional Neural Network (CNN) architectures

This project aims to classify approximately 17,000 blood cell images using the Blood Cells Image Dataset (https://www.kaggle.com/datasets/unclesamulus/blood-cells-image-dataset/data) into 8 classes, using the following pre-trained architectures in TensorFlow:
* MobileNetV2
* EfficientNetB4
* DenseNet201
* InceptionV3
* ResNet50V2

then we evaluate each model by accuracy, with robustness improved using image augmentation and dropout regularization in the prediction layers. The models are trained for 15 epochs with the Adam optimizer with identical prediction layers for control. All models except DenseNet201 show no signs of overfitting with EfficientNetB4 performing the best at 94.53% test accuracy.
