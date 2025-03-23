from math import sqrt

def reward_calc(prev_acc_x, prev_acc_y, acc_x, acc_y, Currlanespd, \
	spd, x, target_x, time_elap = 0.1, time_elap_lane_change = 0, present_B = 0, rel_X_B = 0, rel_Y_B = 0,\
	present_C = 0, rel_X_C = 0, rel_Y_C = 0, present_D = 0,\
	rel_X_D = 0, rel_Y_D = 0, present_E = 0, rel_X_E = 0, rel_Y_E = 0, present_F = 0, rel_X_F = 0, rel_Y_F = 0, \
	present_G = 0, rel_X_G = 0, rel_Y_G = 0, alpha = 1):
	
	#Comfort
	jerk_penalty_x = -alpha*(abs(acc_x-prev_acc_x))/time_elap
	jerk_penalty_y = -alpha*(abs(acc_y-prev_acc_y))/time_elap
	jerk_penalty = jerk_penalty_x + jerk_penalty_y

	#Efficiency
	time_penalty = -time_elap_lane_change
	pos_penalty = -abs(x-target_x)

	#Safety
	col_penalty = 0

	vehicles = [(present_B, rel_X_B, rel_Y_B), (present_C, rel_X_C, rel_Y_C),\
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

		return 0
