#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np, sys
import matplotlib.pyplot as plt

np.random.seed(1)

# In[2]:


data = np.loadtxt('misc/dane2.txt')
X_train = data[:, 0][..., np.newaxis]
y_train = data[:, 1][..., np.newaxis]

X_test = data[:, 0][..., np.newaxis]
y_test = data[:, 1][..., np.newaxis]


# In[3]:


def tanh(x):
    return np.tanh(x)


def tanh2deriv(output):
    return 1 - (output ** 2)


def relu(x):
    return (x > 0).astype(float) * x


def relu2deriv(x): # derived
    return (x > 0).astype(float)


# In[4]:


alpha, iterations, hidden_size = (0.0003, 10000, 100)
input_size, output_size = (1, 1)
batch_size = len(X_train)  # 1

# In[5]:


cum_losses = []
weights_0_1 = 0.02 * np.random.random((input_size, hidden_size)) - 0.01
weights_1_2 = 0.2 * np.random.random((hidden_size, output_size)) - 0.1

# In[6]:


for j in range(iterations):
    cum_loss = 0
    for i in range(int(len(X_train) / batch_size)):
        batch_start, batch_end = ((i * batch_size), ((i + 1) * batch_size))

        # forward pass
        layer_0 = X_train[batch_start:batch_end]
        layer_1 = tanh(np.dot(layer_0, weights_0_1)) # function tanh applied to the layer
        layer_2 = np.dot(layer_1, weights_1_2)

        # backward pass
        layer_2_delta = (y_train[batch_start:batch_end] - layer_2) / batch_size
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * tanh2deriv(layer_1) # function derivation applied to backward layer

        weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
        weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)

        cum_loss += sum(np.abs((y_train[batch_start:batch_end] - layer_2))) / batch_size

    if j % 100 == 1:
        print(cum_loss)
        cum_losses.append(cum_loss)

# In[7]:


plt.plot(cum_losses)
print()

# In[11]:


cum_losses = []
weights_0_1 = 0.02 * np.random.random((input_size, hidden_size)) - 0.01
weights_1_2 = 0.2 * np.random.random((hidden_size, output_size)) - 0.1

# In[12]:


for j in range(iterations):
    cum_loss = 0
    for i in range(int(len(X_train) / batch_size)):
        # forward pass
        batch_start, batch_end = ((i * batch_size), ((i + 1) * batch_size))
        layer_0 = X_train[batch_start:batch_end]
        layer_1 = relu(np.dot(layer_0, weights_0_1))
        layer_2 = np.dot(layer_1, weights_1_2)

        # backward pass
        layer_2_delta = (y_train[batch_start:batch_end] - layer_2) / batch_size
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1)

        weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
        weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)

        cum_loss += sum(np.abs((y_train[batch_start:batch_end] - layer_2))) / batch_size

    if j % 100 == 1:
        print(cum_loss)
        cum_losses.append(cum_loss)

# In[13]:


plt.plot(cum_losses)
plt.show()
