import os
import numpy as np

class PPOMemory:
	def __init__(self, batch_size):
		self.states_A = []
		self.states_B = []
		self.states_C = []
		self.states_D = []
		self.states_E = []
		self.states_F = []
		self.states_G = []
		
		self.probs_acc = [] #Log probs
		self.actions_acc = [] #Actions took
		
		self.probs_steering = [] #Log probs
		self.actions_steering = [] #Actions took
		
		self.vals = [] #Value of critics
		self.rewards = [] #Rewards recieved
		self.dones = []

		self.batch_size = batch_size

	def generate_batches(self):

		print(self.actions_acc)

		n_states = len(self.dones)
		batch_start = np.arange(0, n_states, self.batch_size)
		indices = np.arange(n_states, dtype = np.int64)
		np.random.shuffle(indices)
		batches = [indices[i:i+self.batch_size] for i in batch_start]

		return np.array(self.states_A), np.array(self.states_B), np.array(self.states_C), np.array(self.states_D),\
		np.array(self.states_E),np.array(self.states_F),np.array(self.states_G),\
		np.array(self.actions_steering), np.array(self.probs_steering) ,\
		np.array(self.actions_acc), np.array(self.probs_acc), np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches

	def store_memory(self, state, action_steering, prob_steering, action_acc, prob_acc, vals, reward, done):
		self.states_A.append(state[0])
		self.states_B.append(state[1])
		self.states_C.append(state[2])
		self.states_D.append(state[3])
		self.states_E.append(state[4])
		self.states_F.append(state[5])
		self.states_G.append(state[6])

		self.probs_steering.append(prob_steering)
		self.actions_steering.append(action_steering)
		self.probs_acc.append(prob_acc)
		self.actions_acc.append(action_acc)
		self.vals.append(vals)
		self.rewards.append(reward)
		self.dones.append(done)

	def clear_memory(self):
		self.states_A = []
		self.states_B = []
		self.states_C = []
		self.states_D = []
		self.states_E = []
		self.states_F = []
		self.states_G = []
		
		self.probs_acc = [] #Log probs
		self.actions_acc = [] #Actions took
		
		self.probs_steering = [] #Log probs
		self.actions_steering = [] #Actions took
		
		self.vals = [] #Value of critics
		self.rewards = [] #Rewards recieved
		self.dones = [] 
