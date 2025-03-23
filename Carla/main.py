#Need to work on spawn points

import glob
import os
import sys
import random
import time

import carla
import tensorflow as tf
from tqdm import tqdm
import numpy as np

from ppo_agent.Agent import Agent
from utils import *


SHOW_PREVIEW = False


class CarEnv:
	SHOW_CAM = SHOW_PREVIEW

	def __init__(self):
		self.client = carla.Client('localhost', 2000)
		self.client.set_timeout(10.0)
		self.world = self.client.get_world()
		self.blueprint_library = self.world.get_blueprint_library()
		self.model3 = self.blueprint_library.filter('model3')[0]

		#Need to tweak N
		self.episode = 0
		self.iters = 0
		self.learn_iters = 0
		self.N = 128
		self.episode_rewards = []

		self.agent = Agent(gamma=0.99, alpha=0.0003, gae_lambda=0.95, policy_clip=0.1,\
			batch_size=64, N=2048, n_epochs=10)


	def reset(self):
		self.collision_hist = []
		self.actor_list = []

		flag = False
		while not flag:
			try:
				self.transform = random.choice(self.world.get_map().get_spawn_points())
				self.vehicle = self.world.spawn_actor(self.model3, self.transform)
				flag = True

			except RuntimeError as err:
				print('Collision at spawn point, trying again......')

		self.actor_list.append(self.vehicle)

		self.vehicle.apply_control(carla.VehicleControl(throttle = 0.0, brake = 0.0))

		colsensor = self.blueprint_library.find('sensor.other.collision')
		self.colsensor = self.world.spawn_actor(colsensor, self.transform, attach_to = self.vehicle)

		self.actor_list.append(self.colsensor)
		self.colsensor.listen(lambda event: self.collision_data(event))

		return None

	def generate_scenario(self):
		flag = False
		length = len(self.world.get_map().get_spawn_points())
		while not flag:
			try:
				spawn_list = self.world.get_map().get_spawn_points()
				start_spawn_transform = spawn_list[np.random.choice(length)]
				self.actor =  self.world.spawn_actor(blueprint = self.model3, transform = start_spawn_transform)
				flag = True

			except RuntimeError as err:
				print('Collision at spawn point, trying again......')

		
		return None

	def collision_data(self, event):
		self.collision_hist.append(event)
		return None

	def step(self, acc, steering):
		acc = float(acc)
		steering = float(steering)

		if acc>=0:
			self.vehicle.apply_control(carla.VehicleControl(throttle = acc, steer = steering, brake = 0.0))
		else:
			self.vehicle.apply_control(carla.VehicleControl(brake = abs(acc), steer = steering, throttle = 0.0))

		return None


	def get_state(self):
		location = self.vehicle.get_location()
		loc_x = location.x
		velocity = self.vehicle.get_velocity()
		vX = velocity.x
		vY = velocity.y
		goLeft = False
		goRight = False
		acceleration = self.vehicle.get_acceleration()
		aY = acceleration.y
		vehicleControl = self.vehicle.get_control()
		steerAngle = vehicleControl.steer
		#junction_data =
		#current_lanespd = 
		#
		input_A = [loc_x, vX,vY,steerAngle,aY,goLeft,goRight, -1, 60]
		input_B = [0, 0, 0, 0, 0]
		input_C = [0, 0, 0, 0, 0]
		input_D = [0, 0, 0, 0, 0]
		input_E = [0, 0, 0, 0, 0]
		input_F = [0, 0, 0, 0, 0]
		input_G = [0, 0, 0, 0, 0]

		return self.make_tensor(input_A, input_B, input_C, input_D, input_E, input_F, input_G)

	def make_tensor(self, input_A, input_B, input_C, input_D, input_E, input_F, input_G):

		input_A = tf.convert_to_tensor([input_A])
		input_B = tf.convert_to_tensor([input_B])
		input_C = tf.convert_to_tensor([input_C])
		input_D = tf.convert_to_tensor([input_D])
		input_E = tf.convert_to_tensor([input_E])
		input_F = tf.convert_to_tensor([input_F])
		input_G = tf.convert_to_tensor([input_G])
		
		return input_A, input_B, input_C, input_D, input_E, input_F, input_G


	def is_done(self):
		#Check for some condn
		if self.iters%100==0:
			return True

		return False

	def run_exp(self):
		'''
		reset env
		done = False
		start: 
			get_state
			get_reward
			step
			remember
			learn_condn?
			repeat till done
		'''
		self.reset()
		self.generate_scenario()
		done = False
		self.episode += 1
		episode_reward = 0
		prev_acc_x = 0
		prev_acc_y = 0

		while not done:
			self.iters+=1
			input_A, input_B, input_C, input_D, input_E, input_F, input_G = self.get_state()
			mu_steering, var_steering, steering, mu_acc, var_acc, acc, value = self.agent.choose_action(input_A,\
				input_B, input_C, input_D, input_E, input_F, input_G)

			prob_steering = self.agent.calc_log_prob(mu_steering, var_steering, steering)
			prob_acc = self.agent.calc_log_prob(mu_acc, var_acc, acc)

			self.step(acc, steering)
			spd = input_A[0,2]
			x = input_A[0,0]
			target_x = x

			reward = reward_calc(prev_acc_x, prev_acc_y, 0, acc, 60, spd, x, target_x)
			episode_reward+=reward

			prev_acc_y = acc

			state = [input_A, input_B, input_C, input_D, input_E, input_F, input_G]

			self.agent.remember(state, steering, prob_steering, acc, prob_acc, value, reward, done)

			if self.iters%self.N == 0:
				self.learn_iters+=1
				self.agent.learn()

			done = self.is_done()


		self.episode_rewards.append(episode_reward)
		self.running_reward = np.mean(self.episode_rewards[-100:])

		print(f'Episodes: {self.episode}; Total Steps: {self.iters}; Learn Steps: {self.learn_iters}; Episode Reward = {episode_reward}; Running Reward: {self.running_reward}')

		return None


if __name__ == '__main__':
	env = CarEnv()
	for epoch in tqdm(range(1000)):
		env.run_exp()

	














