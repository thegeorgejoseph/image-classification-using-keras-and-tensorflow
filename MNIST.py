#!/usr/bin/env python
# coding: utf-8

# In[6]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# In[7]:


print(tf.__version__)


# In[9]:


fashion_mnist = keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()


# In[10]:


train_images.shape


# In[12]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
len(train_labels)


# In[13]:


train_labels


# In[14]:


test_images.shape


# In[15]:


len(test_images)


# In[18]:


plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()


# In[19]:


train_images = train_images/255.0
test_images = test_images/255.0


# In[24]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap = plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


# In[26]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


# In[27]:


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[29]:


model.fit(train_images,train_labels,epochs=5)


# In[30]:


model.fit(train_images,train_labels,epochs=25)


# In[32]:


model.fit(train_images,train_labels,epochs=50)


# In[34]:


test_loss ,test_acc = model.evaluate(test_images, test_labels)
print('Test Accuracy = ', test_acc)


# In[35]:


predictions = model.predict(test_images)


# In[36]:


predictions[0]


# In[37]:


np.argmax(predictions[0])


# In[39]:


test_labels[0]


# In[42]:


def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)
  
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)
  
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')



# In[43]:


i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()


# In[44]:


i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()


# In[45]:


num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()


# In[46]:


img = test_images[0]

print(img.shape)


# In[48]:


img = (np.expand_dims(img,0))

print(img.shape)


# In[49]:


predictions_single = model.predict(img)

print(predictions_single)


# In[50]:


plot_value_array(0, predictions_single, test_labels)
plt.xticks(range(10), class_names, rotation=45)
plt.show()


# In[51]:


prediction_result = np.argmax(predictions_single[0])
print(prediction_result)


# In[52]:


test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)


# In[ ]:


t


# In[ ]:




