



import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
#import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

#44 is the number of train steps


def load_dataset():
	"""
	Input :None

	Output: test data generator and train data generator
	"""
    train_data = ImageDataGenerator(horizontal_flip=True,shear_range=0.2,zoom_range=0.2,rescale=1./255)
    test_data = ImageDataGenerator(rescale=1./255)
    train_data = train_data.flow_from_directory('dataset/train_data/',target_size=(64,64),batch_size=32,class_mode='binary')
    #If it is is a multiclass problem , it is better to leave the claas_mode value to default that is categorical
    test_data = test_data.flow_from_directory('dataset/test_data/',target_size=(64,64),batch_size=32,class_mode='binary')
    return train_data,test_data




train_data,test_data =  load_dataset()


def myModel(input_shape):
    """
    Implementation of the myModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    
    X_input = Input(shape = input_shape)
    
    #first conv layer
    X = Conv2D(filters=32,kernel_size=(7,7),strides=(1,1),name="Conv0",padding='same')(X_input)
    X = BatchNormalization(axis = 3 , name="BN0")(X)
    X = MaxPooling2D(name="max_pool0")(X)
    X = Activation('relu')(X)
    X = Dropout(0.25)(X)
    
    
    #second conv layer
    X = Conv2D(filters=32,kernel_size=(7,7),strides=(1,1),name="Conv0",padding='same')(X_input)
    X = BatchNormalization(axis = 3 , name="BN0")(X)
    X = MaxPooling2D(name="max_pool0")(X)
    X = Activation('relu')(X)
    X = Dropout(0.25)(X)
    
    X = Flatten()(X)

    #first hidden layer
    X = Dense(32,activation='relu',name="hidden_layer1")(X)
    #output layer
    X = Dense(1,activation="softmax",name="fully_coonected")(X)
    model = Model(inputs = X_input, outputs = X, name='myModel')
    
    return model



classifier = myModel((64,64,3,))


from keras.metrics import categorical_crossentropy

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# In case of multi class classification use loss function as categorical_crossentropy

classifier.fit_generator(train_data,44,validation_data=test_data,validation_steps=44,epochs=2)


classifier.summary()


preds_train = classifier.evaluate_generator(train_data,44)

names = classifier.metrics_names
print("Train Loss = " + str(preds_train[0]))
print("Train accuracy = " + str(preds_train[1]))

preds_test = classifier.evaluate_generator(test_data,44)

print("Test Loss = " + str(preds_train[0]))
print("Test accuracy = " + str(preds_train[1]))

pred = classifier.predict_generator(test_data,44)

train_data_labels = train_data.class_indices

pred = ['cat' if val<=0.5 else 'dog' for val in pred]





