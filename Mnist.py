#!/usr/bin/env python
# coding: utf-8

# In[1]:


from   keras.datasets  import  mnist


# In[2]:


dataset = mnist.load_data('mnist.db')


# In[3]:


train , test = dataset


# In[4]:


X_train , y_train = train


# In[5]:


X_test , y_test = test


# In[6]:


X_train.shape


# In[7]:


X_test.shape


# In[8]:


X_train = X_train.reshape(-1 , 28*28)
X_test = X_test.reshape(-1 , 28*28)


# In[9]:


X_train.shape


# In[10]:


y_train.shape


# In[11]:


from  keras.utils.np_utils   import  to_categorical


# In[12]:


y_train = to_categorical(y_train)


# In[13]:


from keras.models import Sequential
from keras.layers import Dense


# In[15]:


model.add(Dense(units=512 , input_dim=28*28 , activation='relu'))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=10, activation='softmax'))


# In[16]:


model.summary()


# In[17]:


from keras.optimizers import RMSprop


# In[18]:


model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', 
             metrics=['accuracy']
             )


# In[20]:


model.fit(X_train, y_train, epochs=20)


# In[ ]:




