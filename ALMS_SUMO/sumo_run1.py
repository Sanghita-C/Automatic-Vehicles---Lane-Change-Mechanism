import traci
import datetime
import traci.constants as tc
import pytz
import math
#import datetime

import tensorflow as tf
import numpy as np
from random import randrange
import pandas as pd
from ppo_agent.Agent import Agent
from ppo_agent.Buffer import PPOMemory
from ppo_agent.Networks import NN
from utils import *


def is_done(veh_ID):
        routeLen = len(traci.vehicle.getRoute(veh_ID))
        presentEdgeIdx = traci.vehicle.getRouteIndex(veh_ID)
        if presentEdgeIdx == (routeLen-1):
                return True
        return False


N = 400
batch_size = 20
n_epochs = 4
alpha = 0.0005
n_games = 20

agent = Agent(gamma=0.99, alpha=alpha, gae_lambda=0.95, policy_clip=0.1,\
        batch_size=batch_size, N=N, n_epochs=n_epochs)

import os, sys
if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
else:
        print("ERROR")
        sys.exit("please declare environment variable 'SUMO_HOME'")


sumoCmd = ["sumo-gui", "-c", "ALMS_3Lane_Diversion.sumocfg"]
traci.start(sumoCmd)

n_steps = 1
learn_iters = 0
avg_score = 0
score_history = []

for epoch in range(n_games):
        steps=0
        episode_reward = 0
        prev_acc = 0
        prev_steering = 0
        time_elap = 0
        while traci.simulation.getMinExpectedNumber() > 0:
                done = False
               
                traci.simulationStep();
                currTime = datetime.datetime.now()
                

                vehicles=traci.vehicle.getIDList();

                for i in range(0,len(vehicles)):

                        #Function descriptions
                        #https://sumo.dlr.de/docs/TraCI/Vehicle_Value_Retrieval.html
                        #https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getSpeed

        ## ------------------------------DATA RETRIEVAL----------------------------------------------------## 
                        vehid = vehicles[i]
                        x, y = traci.vehicle.getPosition(vehicles[i])
                        coord = [x, y]
                        #lon, lat = traci.simulation.convertGeo(x, y)
                        #gpscoord = [lon, lat]
                        spd = round(traci.vehicle.getSpeed(vehicles[i]),2)
                        lateralSpd = round(traci.vehicle.getLateralSpeed(vehicles[i]),2)
                        longitudeSpd = math.sqrt(spd**2 - lateralSpd**2) 
                        maxaccel = round(traci.vehicle.getAccel(vehicles[i]),2)
                        presentaccel = round(traci.vehicle.getAcceleration(vehicles[i]),2)
                        steerAngle = round(traci.vehicle.getAngle(vehicles[i]),2)
                        Currlanespd = traci.lane.getMaxSpeed(traci.vehicle.getLaneID(vehicles[i]))
                        numJun = traci.junction.getIDCount()
                        JunList = traci.junction.getIDList()
                        Jpos = [traci.junction.getPosition(j) for j in JunList]
                        goLeft = False
                        goRight = False      

                        input_A = tf.convert_to_tensor([[x, lateralSpd, longitudeSpd, steerAngle, presentaccel, goLeft, goRight, Jpos, Currlanespd]])
                        input_B = tf.convert_to_tensor([[0, 4, 4, 0, 0]])
                        input_C = tf.convert_to_tensor([[0, 4, 4, 0, 0]])
                        input_D = tf.convert_to_tensor([[0, 4, 4, 0, 0]])
                        input_E = tf.convert_to_tensor([[0, 4, 4, 0, 0]])
                        input_F = tf.convert_to_tensor([[0, 4, 4, 0, 0]])
                        input_G = tf.convert_to_tensor([[0, 4, 4, 0, 0]])


                        mu_steering, var_steering, steering, mu_acc, var_acc, acc, value = agent.choose_action([input_A,\
                                input_B, input_C, input_D, input_E, input_F, input_G])

                        
                        traci.vehicle.setaccel(vehicles[i], acc)
                        spd = calc_new_spd(acc, spd)
                        #Need to work on steering and acc

                        reward, fail = reward_calc(prev_acc, acc, prev_steering, steering, time_elap, Currlanespd, spd, x\
                                present_B, rel_X_B, rel_Y_B, present_C, rel_X_C, rel_Y_C,\
                                present_D, rel_X_D, rel_Y_D, present_E, rel_X_E, rel_Y_E,\
                                present_F, rel_X_F, rel_Y_F, present_G, rel_X_G, rel_Y_G)

                        episode_reward+=reward


                        if fail or is_done(vehicles[i]):
                                done  = True



                        state = [input_A, input_B, input_C, input_D, input_E, input_F, input_G]
                        action_steering = steering
                        action_acc = acc
                        prob_steering = agent.calc_log_prob(mu_steering, var_steering, steering)
                        prob_acc = agent.calc_log_prob(mu_acc, var_acc, acc)
                        vals = value

                        agent.remember(state, action_steering, prob_steering, action_acc, prob_acc, vals, reward, done)

                        prev_acc = acc
                        prev_steering = steering

                        if n_steps%N==0:
                                agent.learn()
                                learn_iters+=1







                        ##----------CONTROL Vehicles and Traffic Lights----------##

                        #***SET FUNCTION FOR VEHICLES***
                        #REF: https://sumo.dlr.de/docs/TraCI/Change_Vehicle_State.html
                        if steps==447:
                                traci.vehicle.changeTarget(vehicles[i],'3to6')
                                #  lanechmode = traci.vehicle.getLaneChangeMode(vehicles[i])
                                #  print("lane change mode : ",lanechmode)
                                #  laneidx = traci.vehicle.getLaneIndex(vehicles[i])
                                #  #print("lane id : ",traci.vehicle.getLaneID(vehicles[i]))
                                #  print(currTime)
                                #  print("lane index: ",laneidx)
                                #  traci.vehicle.changeLane(vehicles[i],1,20.0)
                        # if steps == 457:
                        #         print(currTime)
                        #         laneidx = traci.vehicle.getLaneIndex(vehicles[i])
                        #         print("lane index: ",laneidx)
                        
                        # if steps == 467:
                        #         print(currTime)
                        #         laneidx = traci.vehicle.getLaneIndex(vehicles[i])
                        #         print("lane index: ",laneidx)
                                
                                # traci.vehicle.setSpeedMode(vehicles[i],0)
                                # traci.vehicle.setSpeed(vehicles[i],0.2)
                                # #object_list = traci.vehicle.getIDList()
                                # #print(object_list)
                                # traci.vehicle.changeTarget(vehicles[i],'3to6')
                        #NEWSPEED = 15 # value in m/s (15 m/s = 54 km/hr)
                        #if vehicles[i]=='veh1':
                        #        traci.vehicle.setSpeedMode('veh1',0)
                        #        traci.vehicle.setSpeed('veh1',NEWSPEED)


                        #***SET FUNCTION FOR TRAFFIC LIGHTS***
                        #REF: https://sumo.dlr.de/docs/TraCI/Change_Traffic_Lights_State.html
                        #trafficlightduration = [5,37,5,35,6,3]
                        #trafficsignal = ["rrrrrrGGGGgGGGrr", "yyyyyyyyrrrrrrrr", "rrrrrGGGGGGrrrrr", "rrrrryyyyyyrrrrr", "GrrrrrrrrrrGGGGg", "yrrrrrrrrrryyyyy"]
                        #tfl = "cluster_4260917315_5146794610_5146796923_5146796930_5704674780_5704674783_5704674784_5704674787_6589790747_8370171128_8370171143_8427766841_8427766842_8427766845"
                        
                        #traci.trafficlight.setPhaseDuration(tfl, trafficlightduration[randrange(6)])
                        #traci.trafficlight.setRedYellowGreenState(tfl, trafficsignal[randrange(6)])

                        ##------------------------------------------------------##
                        steps+=1
                        n_steps+=1

        score_history.append(episode_reward)
        avg_score = np.mean(score_history[-100:])

        traci.close()


#Generate Excel file
#columnnames = ['dateandtime', 'vehid', 'coord', 'gpscoord', 'spd', 'edge', 'lane', 'displacement', 'turnAngle', 'nextTLS', \
                      # 'tflight', 'tl_state', 'tl_phase_duration', 'tl_lanes_controlled', 'tl_program', 'tl_next_switch']
#dataset = pd.DataFrame(packBigData, index=None, columns=columnnames)
#dataset.to_excel("output.xlsx", index=False)
#time.sleep(5)

