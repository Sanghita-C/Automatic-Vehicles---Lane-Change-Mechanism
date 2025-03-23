import math

import tensorflow as tf
from tensorflow.keras.models import save_model, load_model

from Networks import NN
from Buffer import PPOMemory

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

	def choose_action(self, observation):
		state = tf.convert_to_tensor([observation], dtype=tf.float32)
		mu_steering, var_steering, mu_acc, var_acc = self.actor(
			state)  # dist stands for distribution
		value = self.critic(state)

		steering = tf.random.normal(mean=mu_steering, stddev=var_steering**0.5)
		steering = np.clip(steering, -math.pi/6, math.pi/6)

		acc = tf.random.normal(mean=mu_acc, stddev=var_acc**0.5)
		acc = np.clip(acc, -4, 4)

		mu_steering = tf.squeeze(mu_steering)
		var_steering = tf.squeeze(var_steering)
		steering = tf.squeeze(steering)

		mu_acc = tf.squeeze(mu_acc)
		var_acc = tf.squeeze(var_acc)
		acc = tf.squeeze(acc)

		value = tf.squeeze(value)

		return mu_steering, var_steering, steering, mu_acc, var_acc, acc, value

	def calc_log_prob(self, mu, var, action):
		p1 = -((action-mu)**2)/(2*tf.clip_by_value(var, 1e-3))
		p2 = -tf.math.log(tf.math.sqrt(2*math.pi*var))
		return p1+p2

	def learn(self):
		for epoch in range(self.n_epochs):
			state_arr, action_steering_arr, old_probs_steering_arr, action_acc_arr, old_probs_acc_arr, \
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
				states = tf.convert_to_tensor(
					state_arr[batch], dtype=tf.float32)

				old_probs_steering = tf.convert_to_tensor(
					old_probs_steering_arr[batch], dtype=tf.float32)

				old_probs_acc = tf.convert_to_tensor(
					old_probs_acc_arr[batch], dtype=tf.float32)

				actions_steering = tf.convert_to_tensor(
					list(zip(range(len(batch)),action_steering_arr[batch])), dtype=tf.float32)

				actions_acc = tf.convert_to_tensor(
					list(zip(range(len(batch)), action_acc_arr[batch])), dtype=tf.float32)

				with tf.GradientTape(persistent=True) as tape:

					mu_steering, var_steering, mu_acc, var_acc = self.actor(states)
					
					critic_val = self.critic(states)
					critic_val = tf.squeeze(critic_val)

					new_probs_steering = self.calc_log_prob(mu_steering, var_steering, actions_steering)
					new_probs_acc = self.calc_log_prob(mu_acc, var_acc, actions_acc)

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