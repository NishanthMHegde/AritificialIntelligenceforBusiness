
"""
We build an ANN whih consists of 3 Input node, 64 nodes in first hidden layer
and 32 hidden node in second hidden layer and 5 nodes in the output layer
where each node corresponds to one of the below actions:
1. Decrease temperature by 3 degrees.
2. Decrease temperature by 1.5 degrees
3. Do not transfer any heat.
4. Increase temperature by 3 degrees.
5. Increase temperature by 1.5 degrees
"""

import keras
from keras.layers import Input,Dense
from keras.model import Model 
from keras.optimizers import Adam

class Brain(object):
	def __init__(self, learning_rate=0.005, number_actions=5):
		self.learning_rate = learning_rate
		states = Input(shape=(3,))
		#first hidden layer
		x = Dense(units=64, activation="sigmoid")(states) #create a fully connected system by connecting the Input states
		#second hidden layer
		y = Dense(units=32, activation="sigmoid")(x)
		#create the output layer
		q_values = Dense(units=number_actions, activation="softmax")(y)
		#create the model
		self.model = Model(inputs=states, outputs=q_values)
		#compile the model using Adam optimizer and mean square error loss function
		self.model.compile(loss="mse", optimizer=Adam(lr=learning_rate))