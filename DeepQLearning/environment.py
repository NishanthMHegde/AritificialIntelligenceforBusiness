
#FIRST STEP IS TO DEFINE THE ENVIRONMENT FOR OUR AGENT

import numpy as np 
class Environment(object):

	#initialize all the important parameters
	def __init__(self, initial_month=0,initial_number_of_users=10,initial_data_rate=60,optimal_temperature=(18.0, 24.0)):
		self.initial_data_rate = initial_data_rate
		self.initial_number_of_users = initial_number_of_users
		self.monthly_atmospheric_temperatures = [1.0, 5.0, 7.0, 10.0, 11.0, 20.0, 23.0, 24.0, 22.0, 20.0, 10.0, 5.0, 1.0]
		self.atmospheric_temperature = self.monthly_atmospheric_temperatures[initial_month]
		self.min_temperature = -20
		self.max_temperature = 80
		self.max_number_users = 100
		self.min_number_users = 10
		self.max_update_users = 5
		self.min_data_rate = 20
		self.max_data_rate = 300
		self.max_update_data = 10
		self.current_data_rate = initial_data_rate
		self.current_number_of_users = initial_number_of_users
		self.intrisinic_temperature = self.atmospheric_temperature + 1.25*self.current_number_of_users + 1.25*self.current_data_rate
		self.temperature_ai = self.intrisinic_temperature
		self.temperature_noai = (self.optimal_temperature[0] + self.optimal_temperature[1]) / 2.0
		self.total_enery_ai = 0.0
		self.total_enery_noai = 0.0
		self.reward = 0.0
		self.game_over = 0
		self.train = 1

	#UPDATE THE ENVIRONMENT AS AND WHEN THE AGENT PLAYS AN ACTION
	def update_env(self, direction,energy_ai, month):
		#GETTING THE REWARD
		#calculate the reward for the agent by comparing difference between energy_ai and energy_noai
		energy_noai = 0.0
		if self.temperature_noai < self.optimal_temperature[0]:
			energy_noai = self.optimal_temperature[0] - self.temperature_noai
			self.temperature_noai = self.optimal_temperature[0]
		elif self.temperature_noai > self.optimal_temperature[1]:
			energy_noai = self.temperature_noai - self.optimal_temperature[1]
			self.temperature_noai = self.optimal_temperature[1]
		self.reward = energy_noai - energy_ai
		#scale the reward by multiplying it with 10^-3
		self.reward = 1e-3 * self.reward

		#GETTING THE NEXT STATE
		#calculating the temperature
		self.atmospheric_temperature = self.monthly_atmospheric_temperatures[month]

		#updating the number of users
		self.current_number_of_users = self.current_number_of_users + np.random.randint(-max_update_users, max_update_users)
		if self.current_number_of_users < self.min_number_users:
			self.current_number_of_users = self.min_number_users
		elif self.current_number_of_users > self.max_number_users:
			self.current_number_of_users = self.max_number_users

		#updating the date rate
		self.current_data_rate = self.current_data_rate + np.random.randint(-max_update_data, max_update_data)
		if self.current_data_rate < self.min_data_rate:
			self.current_data_rate = self.min_data_rate
		elif self.current_data_rate > self.max_data_rate:
			self.current_data_rate = self.max_data_rate

		#calculating the Delta of Intrisinic temperature
		past_intrisinic_temperature = self.intrisinic_temperature
		self.intrisinic_temperature = self.atmospheric_temperature + 1.25*self.current_number_of_users + 1.25*self.current_data_rate
		delta_intrisinic_temperature = self.intrisinic_temperature - past_intrisinic_temperature

		#calculating the delta of temperature for AI
		if (direction == -1):
			delta_temperature_ai = -energy_ai
		else:
			delta_temperature_ai = energy_ai

		#Updating the server's current temperature when there is AI
		self.temperature_ai = self.temperature_ai + delta_temperature_ai + delta_intrisinic_temperature
		#Updating the server's current temperature when there is no AI
		self.temperature_noai = self.temperature_noai + delta_intrisinic_temperature

		#GETTING GAME OVER STATE
		if self.temperature_ai < self.optimal_temperature[0]:
			if (self.train == 1):
				self.game_over = 1
			else:
				self.temperature_ai = self.optimal_temperature[0]
				self.total_enery_ai = self.total_enery_ai + self.optimal_temperature[0] - self.temperature_ai
		elif self.temperature_ai > self.optimal_temperature[1]:
			if (self.train == 1):
				self.game_over = 1
			else:
				self.temperature_ai = self.optimal_temperature[1]
				self.total_enery_ai = self.total_enery_ai + self.temperature_ai - self.optimal_temperature[0]
				self.total_enery_ai

		#Updating the scores
		self.total_enery_ai = self.total_enery_ai +  energy_ai
		self.total_enery_noai = self.total_enery_noai +  energy_noai

		#Scaling our next state
		scaled_temperature_ai = (self.temperature_ai - self.min_temperature)/(self.max_temperature - self.min_temperature)
		scaled_number_of_users = (self.current_number_of_users - self.min_number_users)/(self.max_number_users - self.min_number_users)
		scaled_data_rate = (sef.current_data_rate - self.min_data_rate)/(self.max_data_rate - self.min_data_rate)
		next_state = np.matrix([scaled_temperature_ai, scaled_number_of_users, scaled_data_rate])
		return next_state, self.reward, self.game_over

		#RESETTING OUR ENVIRONMENT
		def reset_env(self, new_month):
			self.atmospheric_temperature = self.monthly_atmospheric_temperatures[new_month]
			self.initial_month = self.new_month
			self.current_data_rate = self.initial_data_rate
			self.current_number_of_users = self.initial_number_of_users
			self.intrisinic_temperature = self.atmospheric_temperature + 1.25*self.current_number_of_users + 1.25*self.current_data_rate
			self.temperature_ai = self.intrisinic_temperature
			self.temperature_noai = (self.optimal_temperature[0] + self.optimal_temperature[1]) / 2.0
			self.total_enery_ai = 0.0
			self.total_enery_noai = 0.0
			self.reward = 0.0
			self.game_over = 0
			self.train = 1

		#CREATING AN OBSERVE METHOD WHICH RETURNS THE CURRENT STATE, THE PREVIOUSLY ACQUIRED REWARD AND WHETHER GAME OVER

		def observe(self):
			scaled_temperature_ai = (self.temperature_ai - self.min_temperature)/(self.max_temperature - self.min_temperature)
			scaled_number_of_users = (self.current_number_of_users - self.min_number_users)/(self.max_number_users - self.min_number_users)
			scaled_data_rate = (sef.current_data_rate - self.min_data_rate)/(self.max_data_rate - self.min_data_rate)
			current_state = np.matrix([scaled_temperature_ai, scaled_number_of_users, scaled_data_rate])
			return current_state, self.reward, self.game_over
