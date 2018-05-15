

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
np.random.seed(100)
```

### Step 0 : Define the inputs


```python
def initialize_input_matrices(m,n):
    X = np.random.rand(m*n).reshape([m,n])
    Y = np.random.randint(low=0,high=2,size=m,dtype='int')
    return X,Y
```


```python
def step_0():
    print('Give the dimensions of the input matrix X.',end=" ")
    print('Random Numbers will filled in the matrix of dimension of your choice')
    l = [int(x) for x in input('Number of rows x Number of columns (mxn) ').split(' ')]
    X, Y = initialize_input_matrices(l[0],l[1])
    return X, Y
```

### Step 1 : Initialize the weights and biases with random values


```python
def step_1(m,n):
    wh = np.random.rand(m*n).reshape([n,m])
    bh = np.random.rand(m)
    wout = np.random.rand(m)
    bout = np.random.rand(1)
    return wh,bh,wout,bout
```

### Step 2 : Calculate hidden layer input


```python
def step_2(X,wh,bh):
    return np.dot(X,wh) + bh
```

### Step 3 : Perform non-linear transformation


```python
def sigmoid(z):
    return 1/(1+np.exp(-z))
```


```python
def derivative_sigmoid(z):
#     return sigmoid(z) * (1-sigmoid(z)) # This is the original derivative
    return z*(1-z)
```


```python
def relu(z):
    return np.maximum(z,0)
```


```python
def derivative_relu(z):
    return z * (1-z)
```

### Step 4 : Perform linear and non-linear transformation of hidden layer activation at output layer


```python
def step_4(hiddenlayer_activations,wout,bout):
    temp = np.dot(hiddenlayer_activations, np.transpose(wout))+np.transpose(bout)
    return sigmoid(temp)
```

### Step 5 : Calculate gradient of Error(E) at output layer


```python
def step_5(output,Y):
    return Y-output
```

### Step 6 : Compute slope at output and hidden layer


```python
def step_6(output,hiddenlayer_activations):
    slope_output_layer = derivative_sigmoid(output)
    slope_hidden_layer = derivative_sigmoid(hiddenlayer_activations)
    return slope_output_layer, slope_hidden_layer
```

### Step 7 : Compute delta at output layer


```python
def step_7(E, slope_output_layer, lr):
    return E * slope_output_layer * lr
```

### Step 8 : Calculate Error at hidden layer


```python
def step_8(d_output,wout):
    return np.dot(d_output, np.transpose(wout))
```

### Step 9: Compute delta at hidden layer


```python
def step_9(Error_at_hidden_layer, slope_hidden_layer):
    return np.dot(Error_at_hidden_layer,slope_hidden_layer)
```

### Step 10: Update weight at both output and hidden layer


```python
def step_10(hiddenlayer_activations, d_output,lr,wout,wh,X,d_hiddenlayer):
    wout += np.dot(np.transpose(hiddenlayer_activations),d_output) * lr
    wh += np.dot(np.transpose(X),d_hiddenlayer)*lr
    return wout, wh
```

### Step 11: Update biases at both output and hidden layer


```python
def step_11(bh,bout,d_hiddenlayer,d_output,lr):
    bout += np.sum(d_output,axis=0)*lr
    bh += np.sum(d_hiddenlayer,axis=0)*lr
    return bout,bh
```

### Combining all the steps


```python
def main():
    lr = 0.1
    X, Y = step_0()
    wh, bh, wout, bout = step_1(X.shape[0],X.shape[1])
    hidden_layer_input = step_2(X,wh,bh)
    hiddenlayer_activations = sigmoid(hidden_layer_input)
    output = step_4(hiddenlayer_activations,wout,bout)
    E = step_5(output,Y)
    slope_output_layer, slope_hidden_layer = step_6(output, hiddenlayer_activations)
    d_output = step_7(E,slope_output_layer,lr)
    Error_at_hidden_layer = step_8(d_output,wout)
    d_hiddenlayer = step_9(Error_at_hidden_layer, slope_hidden_layer)
    wout, wh = step_10(hiddenlayer_activations,d_output,lr, wout,wh,X,d_hiddenlayer)
    bout,bh = step_11(bh,bout,d_hiddenlayer,d_output,lr)
    print('X')
    print(X)
    print("Y")
    print(Y)
    print('wh')
    print(wh)
    print('bh')
    print(bh)
    print('wout')
    print(wout)
    print('bout')
    print(bout)
    print('Error')
    print(E)
    print('hidden_layer_input')
    print(hidden_layer_input)
    print('hiddenlayer_activations')
    print(hiddenlayer_activations)
    print('output')
    print(output)
    print('Slope_output_layer')
    print(slope_output_layer)
    print('Slope_hidden_layer')
    print(slope_hidden_layer)
    print('Error_at_hidden_layer')
    print(Error_at_hidden_layer)
    print('d_hiddenlayer')
    print(d_hiddenlayer)
```


```python
main()
```

    Give the dimensions of the input matrix X. Random Numbers will filled in the matrix of dimension of your choice
    Number of rows x Number of columns (mxn) 3 4
    X
    [[0.54340494 0.27836939 0.42451759 0.84477613]
     [0.00471886 0.12156912 0.67074908 0.82585276]
     [0.13670659 0.57509333 0.89132195 0.20920212]]
    Y
    [0 1 1]
    wh
    [[0.00583824 0.30744352 0.95019756]
     [0.12668557 0.07901659 0.31137475]
     [0.63244544 0.69941769 0.64201013]
     [0.92007929 0.29893236 0.5687861 ]]
    bh
    [0.17871631 0.53266264 0.64675778]
    wout
    [0.14157773 0.58091108 0.47861791]
    bout
    [0.38580545]
    Error
    [-0.79792504  0.20575068  0.20591191]
    hidden_layer_input
    [[1.26271416 1.27099058 2.00268557]
     [1.37802391 1.25954909 1.58932365]
     [1.00838588 1.30590541 1.64682526]]
    hiddenlayer_activations
    [[0.77949298 0.78091227 0.88107876]
     [0.79867344 0.77894848 0.83052092]
     [0.73270415 0.78682718 0.83846151]]
    output
    [0.79792504 0.79424932 0.79408809]
    Slope_output_layer
    [0.16124067 0.16341734 0.16351219]
    Slope_hidden_layer
    [[0.17188367 0.1710883  0.10477898]
     [0.16079417 0.17218775 0.14075592]
     [0.19584878 0.16773017 0.1354438 ]]
    Error_at_hidden_layer
    0.001740422877834307
    d_hiddenlayer
    [[0.00029915 0.00029777 0.00018236]
     [0.00027985 0.00029968 0.00024497]
     [0.00034086 0.00029192 0.00023573]]



```python
def with_epochs(num_epochs):
    lr = 0.1
    X, Y = step_0()
    wh, bh, wout, bout = step_1(X.shape[0],X.shape[1])
    
    for i in range(num_epochs):
        hidden_layer_input = step_2(X,wh,bh)
        hiddenlayer_activations = sigmoid(hidden_layer_input)
        output = step_4(hiddenlayer_activations,wout,bout)
        E = step_5(output,Y)

        
        slope_output_layer, slope_hidden_layer = step_6(output, hiddenlayer_activations)
        d_output = step_7(E,slope_output_layer,lr)
        Error_at_hidden_layer = step_8(d_output,wout)
        d_hiddenlayer = step_9(Error_at_hidden_layer, slope_hidden_layer)
        wout, wh = step_10(hiddenlayer_activations,d_output,lr, wout,wh,X,d_hiddenlayer)
        bout,bh = step_11(bh,bout,d_hiddenlayer,d_output,lr)
    
    print('After ' + str(i) + ' number of epochs')
    print('wh')
    print(wh)
    print('bh')
    print(bh)
    print('wout')
    print(wout)
    print('bout')
    print(bout)
    print('Error')
    print(E)
    print('hidden_layer_input')
    print(hidden_layer_input)
    print('hiddenlayer_activations')
    print(hiddenlayer_activations)
    print('output')
    print(output)
    print('Slope_output_layer')
    print(slope_output_layer)
    print('Slope_hidden_layer')
    print(slope_hidden_layer)
    print('Error_at_hidden_layer')
    print(Error_at_hidden_layer)
    print('d_hiddenlayer')
    print(d_hiddenlayer)
```


```python
with_epochs(5)
```

    Give the dimensions of the input matrix X. Random Numbers will filled in the matrix of dimension of your choice
    Number of rows x Number of columns (mxn) 3 4
    After 4 number of epochs
    wh
    [[0.76943692 0.25100514 0.28623503]
     [0.85256579 0.97516081 0.88502684]
     [0.35965401 0.59898881 0.35494212]
     [0.34039293 0.17828602 0.2379152 ]]
    bh
    [0.04531884 0.50585718 0.37672344]
    wout
    [0.59425353 0.63143551 0.14403283]
    bout
    [0.93573781]
    Error
    [0.12376908 0.11403006 0.1209084 ]
    hidden_layer_input
    [[0.89847602 1.28125808 1.02501036]
     [1.42915498 1.56034951 1.34743563]
     [1.26756635 1.14201759 1.05331393]]
    hiddenlayer_activations
    [[0.71063622 0.78266385 0.73594741]
     [0.80676962 0.8264035  0.79371007]
     [0.78032586 0.75804988 0.74141076]]
    output
    [0.87623092 0.88596994 0.8790916 ]
    Slope_output_layer
    [0.10845029 0.1010272  0.10628956]
    Slope_hidden_layer
    [[0.20563238 0.17010115 0.19432882]
     [0.1558924  0.14346076 0.1637344 ]
     [0.17141741 0.18341026 0.19172085]]
    Error_at_hidden_layer
    0.0017090803456094616
    d_hiddenlayer
    [[0.00035144 0.00029072 0.00033212]
     [0.00026643 0.00024519 0.00027984]
     [0.00029297 0.00031346 0.00032767]]

