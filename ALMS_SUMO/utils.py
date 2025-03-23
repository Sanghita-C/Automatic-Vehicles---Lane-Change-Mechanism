from math import sqrt

def reward_calc(prev_acc, acc, prev_steering, steering, time_elap, Currlanespd, \
	spd, x, present_B, rel_X_B, rel_Y_B, present_C, rel_X_C, rel_Y_C, present_D,\
	rel_X_D, rel_Y_D, present_E, rel_X_E, rel_Y_E, present_F, rel_X_F, rel_Y_F, \
	present_G, rel_X_G, rel_Y_G, target_x, alpha = 1, safe_dist):
	
	#Comfort
	jerk_penalty = -alpha*(abs(acc-prev_acc))/time_elap
	#need to add steering jerk

	#Efficiency
	time_penalty = -time_elap
	pos_penalty = -abs(x-target_x)

	#Safety
	col_penalty = 0

	vehicles = [(present_B, rel_X_B, rel_Y_B), (present_C, rel_X_C, rel_Y_C)\
	(present_D, rel_X_D, rel_Y_D), (present_E, rel_X_E, rel_Y_E),\
	(present_F, rel_X_F, rel_Y_F), (present_G, rel_X_G, rel_Y_G)]
	
	for present, rel_x, rel_y in vehicles:
		if present:
			col_penalty += check_collision(rel_x, rel_y, spd)


	#Soft Speed Limit
	speed_penalty = -abs(Currlanespd - spd)

	return jerk_penalty + time_penalty + pos_penalty + col_penalty + speed_penalty


def check_collision(rel_x, rel_y, spd):
	if rel_x ==0 or rel_y == 0:
		return -100

	else:
		dist = sqrt(rel_x**2 + rel_y**2)
		if dist<2*spd:
			return -1/(0.1 + dist)

def calc_new_speed(acc,spd):
	time =0.001
	new_spd = spd + acc*time
	return new_spd
