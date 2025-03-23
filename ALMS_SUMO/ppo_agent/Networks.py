import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
import math

class NN:
	def __init__(self):
		input_A = Input(shape = 9) # x, vx, vy, steering angle, ay, go_left, go_right, junction info, current_lane_speed

		input_B = Input(shape = 5) # Neighbouring vehicle 1 present, rel_x, rel_y, vy, ay (left_lane)
		input_C = Input(shape = 5) # Neighbouring vehicle 2 present, rel_x, rel_y, vy, ay (left_lane)
		input_D = Input(shape = 5) # Neighbouring vehicle 3 present, rel_x, rel_y, vy, ay (centre_lane)
		input_E = Input(shape = 5) # Neighbouring vehicle 4 present, rel_x, rel_y, vy, ay (centre_lane)
		input_F = Input(shape = 5) # Neighbouring vehicle 5 present, rel_x, rel_y, vy, ay (right_lane)
		input_G = Input(shape = 5) # Neighbouring vehicle 6 present, rel_x, rel_y, vy, ay (right_lane)

		left_lane = Concatenate()([input_B, input_C])
		centre_lane = Concatenate()([input_D, input_E])
		right_lane = Concatenate()([input_F, input_G]) 

		ego_lane = Dense(32, activation = 'relu')(input_A)
		left_lane = Dense(32, activation = 'relu')(left_lane)
		centre_lane = Dense(32, activation = 'relu')(centre_lane)
		right_lane = Dense(32, activation = 'relu')(right_lane)

		x = Concatenate()([ego_lane, left_lane, centre_lane, right_lane])
		x = Dense(64, activation = 'relu')(x)

		mu_steering = (math.pi/6)*Dense(1, activation = 'tanh')(x) #radians
		var_steering = Dense(1)(x)
		var_steering = tf.keras.activations.softplus(var_steering)
		
		mu_acc = 4*Dense(1, activation = 'tanh')(x)
		var_acc = Dense(1)(x)
		var_acc = tf.keras.activations.softplus(var_acc)

		self.actor = Model(inputs = [input_A, input_B, input_C, input_D, input_E, input_F, input_G],\
		 outputs = [mu_steering, var_steering, mu_acc, var_acc])
		self.actor.summary()

#---------------------------------------------------------------------------------------------------------------------------------


		input_A = Input(shape = 9) # x, vx, vy, steering angle, ay, go_left, go_right, junction info, curr_lane_Speed

		input_B = Input(shape = 5) # Neighbouring vehicle 1 present, rel_x, rel_y, vy, ay (left_lane)
		input_C = Input(shape = 5) # Neighbouring vehicle 2 present, rel_x, rel_y, vy, ay (left_lane)
		input_D = Input(shape = 5) # Neighbouring vehicle 3 present, rel_x, rel_y, vy, ay (centre_lane)
		input_E = Input(shape = 5) # Neighbouring vehicle 4 present, rel_x, rel_y, vy, ay (centre_lane)
		input_F = Input(shape = 5) # Neighbouring vehicle 5 present, rel_x, rel_y, vy, ay (right_lane)
		input_G = Input(shape = 5) # Neighbouring vehicle 6 present, rel_x, rel_y, vy, ay (right_lane)

		left_lane = Concatenate()([input_B, input_C])
		centre_lane = Concatenate()([input_D, input_E])
		right_lane = Concatenate()([input_F, input_G]) 

		ego_lane = Dense(32, activation = 'relu')(input_A)
		left_lane = Dense(32, activation = 'relu')(left_lane)
		centre_lane = Dense(32, activation = 'relu')(centre_lane)
		right_lane = Dense(32, activation = 'relu')(right_lane)

		x = Concatenate()([ego_lane, left_lane, centre_lane, right_lane])
		x = Dense(64, activation = 'relu')(x)

		critic_val = Dense(1)(x)
		self.critic = Model(inputs = [input_A, input_B, input_C, input_D, input_E, input_F, input_G], outputs = critic_val)
		self.critic.summary()


if __name__ == '__main__':
	nn = NN()
