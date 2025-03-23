import os
import numpy as np

class PPOMemory:
	def __init__(self, batch_size):
		self.states = []
		
		self.probs_acc = [] #Log probs
		self.actions_acc = [] #Actions took
		
		self.probs_steering = [] #Log probs
		self.actions_steering = [] #Actions took
		
		self.vals = [] #Value of critics
		self.rewards = [] #Rewards recieved
		self.dones = []

		self.batch_size = batch_size

	def generate_batches(self):
		n_states = len(self.states)
		batch_start = np.arange(0, n_states, self.batch_size)
		indices = np.arange(n_states, dtype = np.int64)
		np.random.shuffle(indices)
		batches = [indices[i:i+self.batch_size] for i in batch_start]

		return np.array(self.states), np.array(self.actions_steering), np.array(self.probs_steering) ,\
		np.array(self.actions_acc), np.array(self.probs_acc), np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches

	def store_memory(self, state, action_steering, prob_steering, action_acc, prob_acc, vals, reward, done):
		self.states.append(state)
		self.probs_steering.append(prob_steering)
		self.actions_steering.append(action_steering)
		self.probs_acc.append(prob_acc)
		self.actions_acc.append(action_acc)
		self.vals.append(vals)
		self.rewards.append(reward)
		self.dones.append(done)

	def clear_memory(self):
		self.states = []
		
		self.probs_acc = [] #Log probs
		self.actions_acc = [] #Actions took
		
		self.probs_steering = [] #Log probs
		self.actions_steering = [] #Actions took
		
		self.vals = [] #Value of critics
		self.rewards = [] #Rewards recieved
		self.dones = [] 
