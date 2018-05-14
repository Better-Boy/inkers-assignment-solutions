## CONVOLUTION:

Convolution is one of the most important concept in machine learning for image processing. The input image that is used is a 2D image.

Convolution is a process of image recognition where the layers tries to create an outlined image from what it has learned in the past.The more the neurons are exposed to the image , the more better we can recollect the image.

Different convolutional layers are involved in which each layer has different numbers of kernels or filters . The number of filters or kernels depends on ourselves. In Convolution we do not capture a complete image at once. The particular region in the image that is being captured is called local receptive field. We shift the local receptive field across the entire image. So overlapping of the regions of the input image takes place. Refer to the image for better explanation:,
![alt text](https://mlblr.com/images/4-2ConvolutionSmall.gif)


## Feature Engineering
 
Feature engineering is an art of extracting more information from existing data.
Feature engineering can be considered as a sort of transformation from the raw data into the information that the model can interpret or understand.
   
   Predictive power of machine learning algorithms depend on it.
Good feature engineering involves two components. The first is an understanding the properties of the task to be solved and how they might interact with the strengths and limitations of the classifier that is being used.
The second is experimental work where you will be testing your expectations and figuring out what actually works and what doesn't. 

 
 Feature engineering often takes multiple features into considearations as each feature captures different informations and one single feature is never sufficient for the proper functioning of the model.
 
 The process involves 4 steps:
   >1. Brainstorm features: Study the problem at hand,look for possible features and look for the features that you find useful
   >
   > 2. Devise features:Finalise the features that are disticnt,informative and independent.
   > 
   >3. Select features: Use different feature importance scorings and feature selection methods to prepare one or more “views” for your models to operate upon.
   >
   >4. Evaluate models: Estimate model accuracy on unseen data using the chosen features.


## Steps to upload project on github:
 
 * Open the link https://github.com.
 * Once you have created your github account then click on sign in button
 * Enter your username and password
 * After signing in , click on  repository tab and then click on "New" tab.Type the repository name. Foreg: machineLearning and type the description.
 * Select the checkbox "public".
 * Click on create repository.
 * Go to the root of the project which you want to upload the files.
 * Open the git command tool and type " git init ". This initializes the git repository. Press enter.
 * Then write "git add  file name" Eg. git add readme.md. Press enter. This "git add filename" commands helps you us add the files that we want to uplaod.
 * To add all files use 'git add .'
 * Then copy the command "git commit -m "first commit" and paste in git command tool and press enter.
 * To add it to the remote repository type--> git remote add origin
 * Now we will push the project. So write this last sentence "git push -u origin master".
 * Refresh the repo and we can see our uploaded files there. Example record.json.



## Activation Function

In simple words Activation function is a function which takes one or more inputs and converts it to an output. When an activation function is applied over the output of a convolution then it becomes a neuron. Different types of activation functions that can be used are Sigmoid, tanh, ReLU. The most widely used activation function today is ReLU - rectified Linear Unit
The formula for this activation function is given below

f(x)=max(0,x))f(x)=max(0,x))