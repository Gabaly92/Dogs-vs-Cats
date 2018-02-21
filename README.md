# Machine Learning Engineer Nanodegree
# Deep Learning & Computer Vision
## Project: Capstone Project

In this competition, the goal is to use a machine learning algorithm to classify whether images contain either a dog or a cat. 
If we look at this kind of problem from a high-level point of view, we would find that while a simple task like this one is 
easy for humans, on the contrary, it is very difficult for a computer. The purpose of this project is to determine how 
well an AI can get in solving an Asirra(Animal Species Image Recognition for Restricting Access) which is basically a 
Captcha that asks people to identify photographs of cats and dogs to login in a website.

### Install

This project requires **Python 2.xx and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [Torch Vision](http://pytorch.org)
- [Python Imaging Library](http://www.pythonware.com/products/pil/)

### Code
You will need the following files inaddition to Nvidia Titan X GPU to run the project

1) 'model.py'
Used for downloading a pre-trained resnet18 that was already trained on the ImageNet dataset, This is available on the  Torchvision website  The model has 1000 output classes, so to make it suitable for our case we use the same weights of the model and just convert the final layer from 1000 to 2 classes
2) 'metrics.py'
Used for creating the confusion matrix and calculating the accuracy, precision and recall
3) 'Train.py'
Used for doing the actual training on our dataset, the network takes a 224x224 image patch. The network uses the cross-entropy loss function and the stochastic gradient descent algorithm
4) 'Test.py'
Used for testing the network, it takes 5 patches of the input image (4 corners and 1 center) and it computes the output for all these 5 patches and takes the average of them as the final prediction for that image, we used 5 patches so that network is able to see the full image as they are not perfectly 224x224 resolution images

### Data

The dataset consists of diverse images of cats and dogs in different poses, lighting,..etc you can download the training and testing set from https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data

**Target Variable**
- `Image Class`: Probability that the image contains a dog
