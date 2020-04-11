import numpy as np 
#PLEASE READ THE PDF FILE TO CHECK HOW THE GRID OR ENVIRONMENT LOOKS LIKE VISUALLY
#set the gamma and alpha parameters
alpha = 0.9
gamma = 0.75

###########CREATE OUR ENVIRONMENT#############

#our location to state dictionary
location_to_state = {'A':0,
                    'B':1,
                    'C':2,
                    'D':3,
                    'E':4,
                    'F':5,
                    'G':6,
                    'H':7,
                    'I':8,
                    'J':9,
                    'K':10,
                    'L':11,
                    }

#list of actions(possible state transitions) which our agent can make in our maze
actions = [0,1,2,3,4,5,6,7,8,9,10,11]

#Reward matrix
#1 if it is possible to move to that state from current state. 0 if not possible

R = np.array([
        [0,1,0,0,0,0,0,0,0,0,0,0],
        [1,0,1,0,0,1,0,0,0,0,0,0],
        [0,1,0,0,0,0,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,1,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,0,0,0],
        [0,1,0,0,0,0,0,0,0,1,0,0],
        [0,0,1,0,0,0,0,1,0,0,0,0],
        [0,0,0,1,0,0,1,0,0,0,0,1],
        [0,0,0,0,1,0,0,0,0,1,0,0],
        [0,0,0,0,0,1,0,0,1,0,1,0],
        [0,0,0,0,0,0,0,0,0,1,0,1],
        [0,0,0,0,0,0,0,1,0,0,1,0],
        ])

###############Creating our Q-Learning Solution #############



######PUTTING OUR CODE TO PRODUCTION#########
    
#In this step, we take a starting and ending location and print out the best route
#to travel from one state to another state. 

def route(starting_location, ending_location):
    """
    This function will return the best route from starting location to ending 
    location.
    """
    #Q-Learning matrix
    Q = np.array(np.zeros([12, 12]))
    #create a dictionary mapping between state to location
    state_to_location = {state:location for location,state in location_to_state.items()}
    #get the ending state and increase its reward to 1000 in the initial reward
    #matrix
    ending_state = location_to_state[ending_location]
    R_new = np.copy(R)
    R_new[ending_state, ending_state] = 1000
    #take 1000 iterations of our maze and update the value of TD, Q and R

    for i in range(0,1000):
        #randomly select a starting state in our maze of 12 tiles.
        current_state = np.random.randint(0, 12)
        #list of actions which are playable
        playable_actions = []
        #get the ist of states reachable from current state
        for j in range(0, 12):
            if R_new[current_state, j] > 0:
                playable_actions.append(j)
        #randomly select the next state from the list of playable states.
        next_state = np.random.choice(playable_actions)
        #calculate the Temporal Difference between current state and next state
        #here action can be considered the next state
        #Q[current_state, np.argmax(Q[current_state,])] gives the Q-value which is
        #the column which has the highest Q-value belonging to the current state row.
        TD = R_new[current_state, next_state] + (gamma * Q[next_state, np.argmax(Q[next_state,])]) - Q[current_state, next_state]
        #now calculate Q-value of the current state
        Q[current_state, next_state] = Q[current_state, next_state] + alpha * TD
    
    #get the list of states to traverse from starting state to ending state
    starting_state = location_to_state[starting_location]
    ending_state = location_to_state[ending_location]
    next_state = starting_state
    route = [starting_location]
    #loop from starting state to ending state
    while next_state != ending_state:
        next_state = np.argmax(Q[starting_state,])
        next_location = state_to_location[next_state]
        route.append(next_location)
        starting_state = next_state
    return route

#Lets find best route from E to G
print("Lets find best route from E to G")
print(route('E', 'G'))

#Find the best route between 2 states give an intermediatery state
print("Find the best route between 2 states give an intermediatery state")

def route_with_intermediate(starting_location, intermediate_location, ending_location):
    first_route = route(starting_location, intermediate_location)
    second_route = route(intermediate_location, ending_location)[1:]
    final_route = first_route + second_route
    return final_route
print(route_with_intermediate('E', 'K', 'G'))