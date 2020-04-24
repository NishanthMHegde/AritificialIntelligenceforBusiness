#Implement the Deep Q Learning along with the experience replay

import numpy as np 

class DQN(object):
	#initialize the max memory and discount factor
	def __init__(self, max_memory==100, discount=0.9):
		self.max_memory = max_memory
		self.discount = discount
		self.memory = list()

	#populate the memory and remove the first memory if the memory list is full
	def remember(self, transition, game_over):
		self.memory.append([transition, game_over])
		if len(self.memory) > self.max_memory:
			del self.memory[0]

	#method that builds a batch of 10 inputs and 10 targets by extracting 10 transitions from memory
	def get_batch(self,model,batch_size=10):
		len_memory = len(self.memory)
		num_inputs = self.memory[0][0][0] #first state in the numpy matrix returned from envirpnment module
		num_outputs = model.output_shape[-1]
		#create empty input and targets
		inputs = np.zeros((min(len_memory), batch_size), num_inputs)
		targets = np.zeros((min(len_memory), batch_size), num_outputs)
		#populate the inputs and targets by looping in batches
		for i, idx in enumerate(np.random.randint(0, len_memory, size=min(len_memory, batch_size))):
			#get the idx-th transition from memory
			current_state, action, reward, next_state = self.memory[idx][0]
			#to check if game was over
			game_over = self.memory[idx][1]
			inputs[i] = current_state
			targets[i] = model.predict(current_state)[0]
			#calculate the Q-value
			Q_sa = np.max(model.predict(next_state)[0])
			#check if game is over
			if game_over:
				targets[i, action] = reward #if game was over, then the cell in the ith row and the column corresponding to the action taken at the previous state will only have reward added
			else:
				targets[i, action] = reward + self.discount * Q_sa
