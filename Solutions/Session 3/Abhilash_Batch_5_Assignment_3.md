## Abhilash K R
### This markdown file contains Solutions for Assignment 3 belonging to Session 3  

**Activation Function**
Initiation capacities are essential parts of neural systems. Both in Perceptron models or Neural Network models, they take in a few information, add non linearity to the information before giving the yield to the following layer. The terms direct and non-straight can be naturally comprehended by considering them as straight lines and non-straight bends.Without the actuation capacities, the perceptron models that take weighted whole of contributions to each layer or the convolution models that utilize portions are only a heap of layers that perform straight relapse and can just distinguish or model information that is directly perceivable or detachable.

There are many activation functions that are used.Let's see how some of the most used activation functions work.

**Sigmoid Function**

Given a value to sigmoid function, it squashes it and outputs value between 0 and 1. So Sigmoid's output can be treated as probability values aka softmax values. As seen in the curve below, from a certain    negative value, the curve moves towards 0. Likewise for some positive values, the curve becomes flat and remains constant. The gradients at these regions will be 0 that leads to vanishing gradient.
![](https://cdn-images-1.medium.com/max/1600/1*XxxiA0jJvPrHEJHD4z893g.png)

**Rectified Linear Unit** 

ReLu is one of the most commonly used activation function. It takes an input **z** and outputs **0** if **z** is negative, or else gives back the same value,**z**. Though ReLu is not entirely linear, we can see that it has a linear form for values that are positive. 
ReLu is easy to compute. Its curve has 0 value for values below 0 and original value (incrementing linearly) for values above 0 i.e during the forward pass, the activity of neurons that are with negative values will be set to 0. 
<br>
Though ReLu is simple and easy to implement, but also has some drawbacks. Consider the horizontal line again for values below 0. During the backpropagation phase, the gradient for this region will be 0 which means the neurons with this gradient will never be updated. There are other variants of ReLu that try to fix this problem.

**Leaky ReLu**

The problem is with the values lesser than 0 in the ReLu curve. So instead of giving 0 for all negative values, a value other than 0 wouldn't allow the gradient to vanish. So, this can be done using Leaky ReLu:

**F(z) = max(0.1z,z)**

With Leaky ReLu, for the negative values, the slope will not be zero. A slight positive slope of 0.1 for values lesser than 0, so that all the activity of neurons with negative values won't be set to 0.s.

**ELU**

Exponential Linear Unit function is of the form:

![](http://saikatbasak.in/public/img/elu.jpg)

In ELU, instead of giving the negative region a slight slope of 0.1, we have α(exp(x) - 1) which gives mean activations of zero that helps in faster convergance.

**SELU**

Scaled Exponential Linear Units is of the form:
![](https://www.hardikp.com/assets/selu.png)  

SELUs induce self-normalizing properties that makes training deep networks with many layers and their learning robust. Though SELUs can give better results than RELUs, some specific configurations are to be  used to get proper results.

<hr>
<br>

**Normalization?**
Information differs generally relying upon the issue. For instance, in an errand like evaluating the pay of a potential worker, the information may be the representative's name, age, current compensation, expected pay, encounter and so on. Each of these could be picked as information highlights which implies they have an influence in information displaying and expectation. As we can watch, each element's esteem is in an alternate scale. Age may shift between 20 to 80. Pay may fluctuate between 200000 to 2500000 and encounter between 0 to 15. Considering that the models decipher the qualities and act likewise, this doesn't appear to be useful for the model. This is the place standardization comes in. Utilizing standardization, we can convey every one of the highlights to a uniform scale so it's less demanding for the model to advance.

**Batch Normalization**
This is one variation of standardization. In the event that you have any experience running a profound system, you more likely than not utilized the parameter batch_size. Rather than taking each example or datapoint at an opportunity to separate highlights, learn and after that improve, we do it for a bunch of tests. Along these lines the model will prepare and improve a clump of tests rather than each one in turn. The same can be connected for standardization. Rather than applying standardization just at the information layer, we do it at each other layer too in the system. This is finished by taking the contribution to a neuron over a group of tests, their mean and fluctuation are calculated and afterward the standardization is connected. Along these lines, the contribution to the non straight activation capacities will be institutionalized and standardized. Standardization for the shrouded/halfway layers is done before applying the activation work. This lessens preparing time and permits higher learning rates with the goal that the model can join sooner.

**Weight Normalization**
The main difference being, while batch normalization considers the mean and variance of a batch of layers for a batch of samples, weight normalization takes the magnitude of weights of each neuron in a layer. Since it acts on each layer independently, the results are more deterministic than batch normalization.

**Layer Normalization**
Though batch norm has led to improved results, specially in feed forward networks, they fail in cases where batch size is 1 or when the data samples are fed one after the other in an online mode. Unlike batch normalization, layer normalization is applied for every sample, one at a time. This works better than the former when batch size is 1. 

Despite the fact that batch normalization has prompted enhanced outcomes, extraordinarily in feed forward neural networks, they bomb in situations where batch size is 1 or when the information tests are fed in a steady progression in an online mode. Not at all like batch normalization, layer normalization is connected for each example, each one in turn. This works superior to anything the previous when batch size is 1.


**Dropout**
Dropout is one of the methods for applying regularization. Neural systems are all inclusive approximators. So sufficiently given information and parameters, the systems can take in any capacity for some known information. In spite of the fact that this may sound effective, it accompanies its own issues. On the off chance that the model is given every one of the information and parameters it needs to take in a capacity, odds are that it does ineffectively on unseen information. This prompts overfitting and poor exactness (or accuracy) comes about. Furthermore, that is the reason it's not a bad plan to deliberately bring some sort of unlearning into the systems. Dropout does precisely that.

<p align="center"><img src="https://mlblr.com/images/dropout.gif"/></p>


Dropout when connected on a layer, deactivates a portion of the layer's neurons. The degree to which the neurons are deactivated relies upon the *p* value assigned to the dropout. For instance, when p = 0.5, half of the neurons from that layer are disabled. These neurons are haphazardly picked which implies each time an alternate arrangement of neurons are chosen relying upon the measure of dropout. By deactivating an arrangement of neurons from a layer, we are reducing the learning by some sum as well as compelling whatever is left of the neurons to learn better. Both of these add to better testing precision. Despite the fact that executing dropout has indicated upgrades in exactness.