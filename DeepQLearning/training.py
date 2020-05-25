import random as rn 
import os
import numpy as np 
from brain import Brain
from dqn import DQN 
from environment import Environment

#Create seeds for python random and numpy random 
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

#set the variables for actions and declare objects
number_actions = 5
number_epochs = 100
epsilon = 0.3
action_boundary = int((number_actions -1)/2)
batch_size = 512
max_memory = 3000
learning_rate = 0.00001
temperature_step = 1.5 #perform actions by either incrementing or decrementing temperature by 1.5 degree celcius
discount = 0.9 

#initialize the objects 
env = Environment(initial_month=0,initial_number_of_users=20,initial_data_rate=30,optimal_temperature=(18.0, 24.0))
brain = Brain(learning_rate=learning_rate, number_actions=number_actions)
dqn = DQN(max_memory=max_memory, discount=discount)

#initialize the trianing params

train = True 
env.train = train
model = brain.model 

#start the training 
if (env.train):
    for epoch in range(1, number_epochs):
        loss = 0.0
        total_reward = 0
        new_month = np.random.randint(0, 12) #randomly choose a month to train 
        env.reset_env(new_month=new_month) #reset the environment for the new month 
        game_over = False 
        current_state, _, _ = env.observe()
        timestep = 0

        #run the trainng for current epoch for every minute in the month for 5 months. In a month there are 30 * 24 * 60 minutes.
        while((not game_over) and (timestep <= 5*30*24*60)):
            #pick a random number between 0 and 1 and see if it is less than epsilon
            #we want to run EXPLORATION for 30% of the time and checking if num is less than 0.3 is a shortcut to achieve it
            if np.random.rand() <= epsilon:
                #do exploration
                action = np.random.randint(0, number_actions)
                #check if direction is negative or positive
                if (action - action_boundary) > 0:
                    direction = 1
                else:
                    direction = -1
                energy_ai = abs(action - action_boundary) * temperature_step
            else:
                #Do inference 
                q_values = model.predict(current_state)
                action = np.argmax(q_values[0]) #actual q_values are present in first column of q_values array
                #check if direction is negative or positive
                if (action - action_boundary) > 0:
                    direction = 1
                else:
                    direction = -1
                energy_ai = abs(action - action_boundary) * temperature_step

            #update the environment with the new energy_ai and direction 
            next_state, reward, game_over = env.update_env(direction,energy_ai, int(timestep/(30*24*60)))
            total_reward = total_reward + reward
            #add the transition and game_over to the memory 
            #transition is the ist of [current_state, action, reward, next_state]
            dqn.remember([current_state, action, reward, next_state], game_over)
            #get the batch of inputs and targets 
            inputs, targets = dqn.get_batch(model,batch_size=batch_size)
            #Compute the loss over the inputs and targets by training the model on the inputs and targets
            loss = loss + model.train_on_batch(inputs, targets)
            timestep = timestep + 1
            current_state = next_state

        #print the statistcs for each Epoch 
        print("\n")
        print("Epoch: {:03d}/{:03d}".format(epoch, number_epochs))
        print("Total Energy spent with an AI: {:.0f}".format(env.total_energy_ai))
        print("Total Energy spent with no AI: {:.0f}".format(env.total_energy_noai))
        model.save("model.h5")
