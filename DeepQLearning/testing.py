import random as rn 
import os
import numpy as np 
from brain import Brain
from dqn import DQN 
from environment import Environment
from keras.models import load_model 

#Create seeds for python random and numpy random 
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

#set the variables for actions and declare objects
#declare only variables related inference mode and not exploration
number_actions = 5
action_boundary = int((number_actions -1)/2)

temperature_step = 1.5 #perform actions by either incrementing or decrementing temperature by 1.5 degree celcius

#initialize the objects 
env = Environment(initial_month=0,initial_number_of_users=20,initial_data_rate=30,optimal_temperature=(18.0, 24.0))

#initialize the trianing params

train = False #We will be testing now  
env.train = train
model = load_model('model.h5') #Load the model we had saved during training  
current_state, _, _ = env.observe()

#start the testing for every minute of the calendar year
for timestep in range(0, 12*30*24*60):

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
    current_state = next_state

#print the statistcs for each Epoch 
print("\n")
print("Total Energy spent with an AI: {:.0f}".format(env.total_energy_ai))
print("Total Energy spent with no AI: {:.0f}".format(env.total_energy_noai))
print("Total energy saved with the AI: {:.0f}".format((env.total_energy_noai-env.total_energy_ai)/env.total_energy_noai * 100))
