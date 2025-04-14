import math

import tensorflow as tf
from tensorflow.keras.models import save_model, load_model
import tensorflow_probability.distributions.Normal as Normal
import numpy as np

from ppo_agent.Networks import NN
from ppo_agent.Buffer import PPOMemory

class Agent:
	def __init__(self, gamma=0.99, alpha=0.0003, gae_lambda=0.95, policy_clip=0.2,
				 batch_size=64, N=2048, n_epochs=10):
		#N is the horizon the number of steps after we do an update
		self.gamma = gamma
		self.policy_clip = policy_clip
		self.n_epochs = n_epochs
		self.gae_lambda = gae_lambda

		self.nn = NN()
		self.actor = self.nn.actor
		self.critic = self.nn.critic
		self.optimizer_critic = tf.keras.optimizers.Adam(lr = alpha)
		self.optimizer_actor = tf.keras.optimizers.Adam(lr = alpha)
		self.memory = PPOMemory(batch_size)

	def remember(self, state, action_steering, prob_steering, action_acc, prob_acc, vals, reward, done):
		self.memory.store_memory(state, action_steering, prob_steering, action_acc, prob_acc, vals, reward, done)

	def save_models(self):
		print('Saving Models')
		save_model(self.actor, r'\Models\actor.h5')
		save_model(self.critic, r'\Models\critic.h5')

	def load_models(self):
		print('Loading Models')
		self.actor = load_model(r'\Models\actor.h5')
		self.critic = load_model(r'\Models\critic.h5')

	def choose_action(self, input_A, input_B, input_C, input_D, input_E, input_F, input_G):
		state = [input_A, input_B, input_C, input_D, input_E, input_F, input_G]
		mu_steering, var_steering, mu_acc, var_acc = self.actor(state)  # dist stands for distribution
		# so the action values are mu_steer, var_steer, mu_acc, var_acc --- this represents
		value = self.critic(state)

		dist_steering = Normal(loc = mu_steering, scale = var_steering**0.5)
		steering = dist_steering.sample()
		# mu_steering + (var_steering**0.5)*tf.random.normal(tf.shape(mu_steering), 0, 1, dtype = tf.float32)
		# tf.random.normal(shape = (1,), mean=mu_steering, stddev=var_steering**0.5)
		# print(steering)
		steering = np.clip(steering, -1.0, 1.0)

		# so this gives us the final steering value 

		dist_acc = Normal(loc = mu_acc, scale = var_acc**0.5)
		acc = dist_acc.sample()
		# print(acc)
		acc = np.clip(acc, -1.0, 1.0)

		# so this gives us the final acceleration value 

		mu_steering = tf.squeeze(mu_steering)
		var_steering = tf.squeeze(var_steering)
		steering = tf.squeeze(steering)
		
		mu_acc = tf.squeeze(mu_acc)
		var_acc = tf.squeeze(var_acc)
		acc = tf.squeeze(acc)

		value = tf.squeeze(value)

		# we will be returning not just the final acc and steering but also the mu and var - we will store these values
		

		return mu_steering, var_steering, steering, mu_acc, var_acc, acc, value

	def calc_log_prob(self, mu, var, action):
		dist = Normal(loc = mu, scale = var**0.5)
		return dist.log_prob(action)

	def learn(self):
		for epoch in range(self.n_epochs):
			state_arr_A, state_arr_B, state_arr_C, state_arr_D, state_arr_E, state_arr_F, state_arr_G,\
			action_steering_arr, old_probs_steering_arr, action_acc_arr, old_probs_acc_arr, \
				vals_arr, reward_arr, done_arr, batches = self.memory.generate_batches()

			values = vals_arr
			advantage = np.zeros(len(reward_arr), dtype=np.float32)

			for t in range(len(reward_arr)-1):
				discount = 1
				a_t = 0

				for k in range(t, len(reward_arr)-1):
					a_t += discount*(reward_arr[k] + self.gamma*values[k+1] *
									 (1-int(done_arr[k])) - values[k])

					discount *= self.gamma*self.gae_lambda

				advantage[t] = a_t

			advantage = tf.convert_to_tensor(advantage)
			values = tf.convert_to_tensor(values)

			for batch in batches:

				states_A = tf.convert_to_tensor(
					[state_arr_A[batch]], dtype=tf.float32)

				states_B = tf.convert_to_tensor(
					[state_arr_B[batch]], dtype=tf.float32)
				
				states_C = tf.convert_to_tensor(
					[state_arr_C[batch]], dtype=tf.float32)
				
				states_D = tf.convert_to_tensor(
					[state_arr_D[batch]], dtype=tf.float32)
				
				states_E = tf.convert_to_tensor(
					[state_arr_E[batch]], dtype=tf.float32)
				
				states_F = tf.convert_to_tensor(
					[state_arr_F[batch]], dtype=tf.float32)

				states_G = tf.convert_to_tensor(
					[state_arr_G[batch]], dtype=tf.float32)

				old_probs_steering = tf.convert_to_tensor(
					old_probs_steering_arr[batch], dtype=tf.float32)

				old_probs_acc = tf.convert_to_tensor(
					old_probs_acc_arr[batch], dtype=tf.float32)

				actions_steering = tf.convert_to_tensor(
					list(zip(range(len(batch)),action_steering_arr[batch])), dtype=tf.float32)

				actions_acc = tf.convert_to_tensor(
					list(zip(range(len(batch)), action_acc_arr[batch])), dtype=tf.float32)

				with tf.GradientTape(persistent=True) as tape:

					mu_steering, var_steering, mu_acc, var_acc = self.actor([states_A, states_B, states_C, states_D, states_E, states_F, states_G])
					
					critic_val = self.critic([states_A, states_B, states_C, states_D, states_E, states_F, states_G])
					critic_val = tf.squeeze(critic_val)

					mu_steering = tf.squeeze(mu_steering)
					mu_acc = tf.squeeze(mu_acc)
					var_steering = tf.squeeze(var_steering)
					var_acc = tf.squeeze(var_acc)

					new_probs_steering = self.calc_log_prob(mu_steering, var_steering, actions_steering[:,1])
					new_probs_acc = self.calc_log_prob(mu_acc, var_acc, actions_acc[:,1])

					probs_ratio_steering = tf.math.exp(new_probs_steering-old_probs_steering)
					probs_ratio_acc = tf.math.exp(new_probs_acc-old_probs_acc)

					weighted_probs_steering = tf.gather(advantage, batch)*probs_ratio_steering
					weighted_probs_acc = tf.gather(advantage, batch)*probs_ratio_acc

					weighted_clipped_probs_steering = tf.math.multiply(tf.clip_by_value
										(probs_ratio_steering, 1-self.policy_clip, 1+self.policy_clip), tf.gather(advantage, batch))

					weighted_clipped_probs_acc = tf.math.multiply(tf.clip_by_value
										(probs_ratio_acc, 1-self.policy_clip, 1+self.policy_clip), tf.gather(advantage, batch))

					actor_loss_steering = tf.reduce_mean(-tf.math.minimum(weighted_probs_steering, weighted_clipped_probs_steering))
					actor_loss_acc = tf.reduce_mean(-tf.math.minimum(weighted_probs_acc, weighted_clipped_probs_acc))

					returns = tf.gather(advantage, batch) + tf.gather(values, batch)
					critic_loss = tf.reduce_mean((returns-critic_val)**2)

					total_loss = actor_loss_acc + actor_loss_steering

				grads_actor = tape.gradient(total_loss, self.actor.trainable_variables)
				self.optimizer_actor.apply_gradients(zip(grads_actor, self.actor.trainable_variables))

				grads_critic = tape.gradient(critic_loss, self.critic.trainable_variables)
				self.optimizer_critic.apply_gradients(zip(grads_critic, self.critic.trainable_variables))

		self.memory.clear_memory()
