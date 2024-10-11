#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 16:49:05 2023

@author: Shreyas S R FINAL only two
"""


import numpy as np
import nashpy as nash
from copy import deepcopy
import math
import random
import time
import matplotlib.pyplot as plt


np.random.seed(200)
random.seed(200)
 
states = 10
actions = 5  
discount = 0.6
max_iterations = 1000

total_avgs = 50


mmql_output = np.zeros((total_avgs,max_iterations))
ql_output = np.zeros((total_avgs,max_iterations))
sorql_output = np.zeros((total_avgs,max_iterations))

P = np.zeros((actions,actions,states,states))

R = np.random.random((actions,actions,states))

for a1 in range(actions):
    for a2 in range(actions):
        for s in range(states):
            P[a1,a2,s] = np.random.random(states)
            # P[a1][a2][s][s] = states #check this. 
            P[a1,a2,s] = P[a1,a2,s] / P[a1,a2,s].sum()
            
#Value Iteration Starts

V = np.zeros(states) #Initial value
while True:
    Q = np.zeros((actions,actions,states))
    for a1 in range(actions):
        for a2 in range(actions):
            Q[a1,a2] = R[a1,a2] + discount * P[a1,a2].dot(V)

    v_prev = deepcopy(V)
    #print(v_prev)
    for s in range(states):
        #print(Q[:,:,s])
        rps = nash.Game(Q[:,:,s])
        #print(rps)
        eqs = rps.lemke_howson(1)
        #print(list(eqs))
        V[s] = rps[list(eqs)][0]
        #print(rps[list(eqs)])

    #print(v_prev)
    #print(V)
    #print(v_prev)
    #print(np.linalg.norm(V-v_prev))

    if np.linalg.norm(V-v_prev) < 0.000001:
        break

# Standard minimax Q-learning

store_avgs = np.zeros((total_avgs,1))
store_time = np.zeros((total_avgs,1))
P_update_time = max_iterations/10

for avg in range(total_avgs):
    starttime = time.time()

    Q = np.random.rand(states,actions,actions)
    state = np.random.randint(0, states)
    
    tot_count = np.zeros((actions,actions,states,states))

    

    for n in range(max_iterations):

        if (n % 100) == 0:
            state = np.random.randint(0, states)

        act1 = random.randint(0,actions-1)
        act2 = random.randint(0,actions-1)


        p_s_new = np.random.random()
        p = 0
        s_new = -1
        while (p < p_s_new) and (s_new < (states - 1)):
            s_new = s_new + 1
            #print(a1,a2,s,s_new)
            p = p + P[act1][act2][state][s_new]

        r = R[act1][act2][state]

        #print(Q[s_new,:,:])

        tot_count[act1][act2][state][s_new] += 1 


        rps = nash.Game(Q[state,:,:])
        #print(rps)
        eqs = rps.lemke_howson(0)
        current_state_value = rps[list(eqs)][0]


        rps = nash.Game(Q[s_new,:,:])
        #print(rps)
        eqs = rps.lemke_howson(0)
        next_state_value = rps[list(eqs)][0]

        delta = r + discount * next_state_value - Q[state, act1,act2]
        #delta = w *(r + discount* next_state_value) + (1-w)*current_state_value - Q[state, act1,act2]
        #dQ = (1 / math.sqrt(np.sum(tot_count[act1][act2][state]))) * delta  
        
        #dQ = (1 / (np.sum(tot_count[act1][act2][state]))**0.501) * delta  
        dQ = (100 /( n+100)) * delta  
        #dQ = beta * delta  
#         dQ = 0.9 * delta 
        Q[state, act1,act2] += dQ
        state = s_new
       
    minimax_Q = np.zeros(states)
    for i in range(states):
        rps = nash.Game(Q[i,:,:])
        #print(rps)
        eqs = rps.lemke_howson(0)
#         print(list(eqs))
        minimax_Q[i] =  rps[list(eqs)][0]
    
    store_avgs[avg][0] = np.linalg.norm(V - minimax_Q)
    endtime = time.time()
    store_time[avg][0] = endtime - starttime 
#     print(np.linalg.norm(V - sor_minimax_Q))    

# y1 = ql_output.mean(0)

# np.savetxt('y1.txt',y1, fmt='%d')
print(np.average(store_avgs),np.std(store_avgs))    
print('Standard Minimax Q-learning took {} seconds',np.average(store_time))
print("------------------------------------------------")



# Two-step minimax Q-learning



store_avgs = np.zeros((total_avgs,1))
store_time = np.zeros((total_avgs,1))
P_update_time = max_iterations/10
for avg in range(total_avgs):
    starttime = time.time()
    Q = np.random.rand(states,actions,actions)
    state = np.random.randint(0, states)
    
    tot_count = np.zeros((actions,actions,states,states))

    

    for n in range(max_iterations):

        if (n % 100) == 0:
            state = np.random.randint(0, states)

        act1 = random.randint(0,actions-1)
        act2 = random.randint(0,actions-1)


        p_s_new = np.random.random()
        p = 0
        s_new = -1
        while (p < p_s_new) and (s_new < (states - 1)):
            s_new = s_new + 1
            #print(a1,a2,s,s_new)
            p = p + P[act1][act2][state][s_new]

        r = R[act1][act2][state]

        #print(Q[s_new,:,:])

        tot_count[act1][act2][state][s_new] += 1 


        rps = nash.Game(Q[state,:,:])
        #print(rps)
        eqs = rps.lemke_howson(0)
        current_state_value = rps[list(eqs)][0]


        rps = nash.Game(Q[s_new,:,:])
        #print(rps)
        eqs = rps.lemke_howson(0)
        next_state_value = rps[list(eqs)][0]
        
        act3 = random.randint(0,actions-1)
        act4 = random.randint(0,actions-1)

        p_s_new = np.random.random()
        p = 0
        s_new1 = -1
        while (p < p_s_new) and (s_new1 < (states - 1)):
            s_new1 = s_new1 + 1
            #print(a1,a2,s,s_new)
            p = p + P[act3][act4][s_new][s_new1]
            
        r1 = R[act3][act4][s_new]  
        
        rps = nash.Game(Q[s_new1,:,:])
        #print(rps)
        eqs = rps.lemke_howson(0)
        next_state_value1 = rps[list(eqs)][0]
        
        
        # delta = (r + discount * next_state_value + discount * (1/ (np.log(n+2)) ) - Q[state, act1,act2])

        # delta = (r + discount * next_state_value + discount * (1/ (np.log(n+2)) ) * (r1 + discount * next_state_value1)- Q[state, act1,act2])
        delta = (r + discount * next_state_value + discount * (80/ (n+80) ) * (r1 + discount * next_state_value1)- Q[state, act1,act2]) 
        #dQ = (1 / math.sqrt(np.sum(tot_count[act1][act2][state]))) * delta 
        #dQ = (1 / (np.sum(tot_count[act1][act2][state]))**0.501) * delta 
        dQ = (100 /( n+100)) * delta  
        #dQ = beta * delta 
#         dQ = 0.9 * delta 
        Q[state, act1,act2] += dQ
        state = s_new1
        
    sor_minimax_Q = np.zeros(states)
    for i in range(states):
        rps = nash.Game(Q[i,:,:])
        #print(rps)
        eqs = rps.lemke_howson(0)
#         print(list(eqs))
        sor_minimax_Q[i] =  rps[list(eqs)][0]
        
    endtime = time.time()
    store_avgs[avg][0] = np.linalg.norm(V - sor_minimax_Q)
    store_time[avg][0] = endtime - starttime 
#     print(np.linalg.norm(V - sor_minimax_Q))    
# y3 = mmql_output.mean(0)
# np.savetxt('y3.txt',y3, fmt='%d')
print(np.average(store_avgs),np.std(store_avgs))    
print('Multi-step minimax Q-learning took {} seconds',np.mean(store_time))

print("------------------------------------------------")



# Three-step minimax Q-learning



store_avgs = np.zeros((total_avgs,1))
store_time = np.zeros((total_avgs,1))
P_update_time = max_iterations/10
for avg in range(total_avgs):
    starttime = time.time()
    Q = np.random.rand(states,actions,actions)
    state = np.random.randint(0, states)
    
    tot_count = np.zeros((actions,actions,states,states))

    

    for n in range(max_iterations):

        if (n % 100) == 0:
            state = np.random.randint(0, states)

        act1 = random.randint(0,actions-1)
        act2 = random.randint(0,actions-1)


        p_s_new = np.random.random()
        p = 0
        s_new = -1
        while (p < p_s_new) and (s_new < (states - 1)):
            s_new = s_new + 1
            #print(a1,a2,s,s_new)
            p = p + P[act1][act2][state][s_new]

        r = R[act1][act2][state]

        #print(Q[s_new,:,:])

        tot_count[act1][act2][state][s_new] += 1 


        rps = nash.Game(Q[state,:,:])
        #print(rps)
        eqs = rps.lemke_howson(0)
        current_state_value = rps[list(eqs)][0]


        rps = nash.Game(Q[s_new,:,:])
        #print(rps)
        eqs = rps.lemke_howson(0)
        next_state_value = rps[list(eqs)][0]
        
        act3 = random.randint(0,actions-1)
        act4 = random.randint(0,actions-1)

        p_s_new = np.random.random()
        p = 0
        s_new1 = -1
        while (p < p_s_new) and (s_new1 < (states - 1)):
            s_new1 = s_new1 + 1
            #print(a1,a2,s,s_new)
            p = p + P[act3][act4][s_new][s_new1]
            
        r1 = R[act3][act4][s_new]  
        
        rps = nash.Game(Q[s_new1,:,:])
        #print(rps)
        eqs = rps.lemke_howson(0)
        next_state_value1 = rps[list(eqs)][0]
        
        
        act5 = random.randint(0,actions-1)
        act6 = random.randint(0,actions-1)
        
        

        p_s_new = np.random.random()
        p = 0
        s_new2 = -1
        while (p < p_s_new) and (s_new2 < (states - 1)):
            s_new2 = s_new2 + 1
            #print(a1,a2,s,s_new)
            p = p + P[act5][act6][s_new1][s_new2]
            
        r2 = R[act5][act6][s_new1]  
        
        rps = nash.Game(Q[s_new2,:,:])
        #print(rps)
        eqs = rps.lemke_howson(0)
        next_state_value2 = rps[list(eqs)][0]
        
        
        # delta = (r + discount * next_state_value + discount * (1/ (np.log(n+2)) ) - Q[state, act1,act2])

        # delta = (r + discount * next_state_value + discount * (1/ (np.log(n+2)) ) * (r1 + discount * next_state_value1)- Q[state, act1,act2])
        delta = (r + discount * next_state_value + discount * (80/ (n+80) ) * (r1 + discount * next_state_value1) + discount**2 *  (10/ (n**2+10))*(r2 + discount * next_state_value2)- Q[state, act1,act2]) 
        #dQ = (1 / math.sqrt(np.sum(tot_count[act1][act2][state]))) * delta 
        #dQ = (1 / (np.sum(tot_count[act1][act2][state]))**0.501) * delta  
        dQ = (100 /( n+100)) * delta  
        #dQ = beta * delta 
#         dQ = 0.9 * delta 
        Q[state, act1,act2] += dQ
        state = s_new2
        """
        sor_minimax_Q3 = np.zeros(states)
        for i in range(states):
            rps = nash.Game(Q[i,:,:])
            #print(rps)
            eqs = rps.lemke_howson(0)
    #         print(list(eqs))
            sor_minimax_Q3[i] =  rps[list(eqs)][0]
        mmql_output[avg][n] = np.linalg.norm(V - sor_minimax_Q3)  
        """
    sor_minimax_Q = np.zeros(states)
    for i in range(states):
        rps = nash.Game(Q[i,:,:])
        #print(rps)
        eqs = rps.lemke_howson(0)
#         print(list(eqs))
        sor_minimax_Q[i] =  rps[list(eqs)][0]
        
    endtime = time.time()
    store_avgs[avg][0] = np.linalg.norm(V - sor_minimax_Q)
    store_time[avg][0] = endtime - starttime 
#     print(np.linalg.norm(V - sor_minimax_Q))    
# y3 = mmql_output.mean(0)
# np.savetxt('y3.txt',y3, fmt='%d')
print(np.average(store_avgs),np.std(store_avgs))    
print('Three-step minimax Q-learning took {} seconds',np.mean(store_time))

print("------------------------------------------------")

# Four-step minimax Q-learning



store_avgs = np.zeros((total_avgs,1))
store_time = np.zeros((total_avgs,1))
P_update_time = max_iterations/10
for avg in range(total_avgs):
    starttime = time.time()
    Q = np.random.rand(states,actions,actions)
    state = np.random.randint(0, states)
    
    tot_count = np.zeros((actions,actions,states,states))

    

    for n in range(max_iterations):

        if (n % 100) == 0:
            state = np.random.randint(0, states)
            
        ###############################################################
        act1 = random.randint(0,actions-1)
        act2 = random.randint(0,actions-1)


        p_s_new = np.random.random()
        p = 0
        s_new = -1
        while (p < p_s_new) and (s_new < (states - 1)):
            s_new = s_new + 1
            #print(a1,a2,s,s_new)
            p = p + P[act1][act2][state][s_new]

        r = R[act1][act2][state]

        #print(Q[s_new,:,:])

        tot_count[act1][act2][state][s_new] += 1 


        rps = nash.Game(Q[state,:,:])
        #print(rps)
        eqs = rps.lemke_howson(0)
        current_state_value = rps[list(eqs)][0]


        rps = nash.Game(Q[s_new,:,:])
        #print(rps)
        eqs = rps.lemke_howson(0)
        next_state_value = rps[list(eqs)][0]
        
        ##############################################################
        act3 = random.randint(0,actions-1)
        act4 = random.randint(0,actions-1)

        p_s_new = np.random.random()
        p = 0
        s_new1 = -1
        while (p < p_s_new) and (s_new1 < (states - 1)):
            s_new1 = s_new1 + 1
            #print(a1,a2,s,s_new)
            p = p + P[act3][act4][s_new][s_new1]
            
        r1 = R[act3][act4][s_new]  
        
        rps = nash.Game(Q[s_new1,:,:])
        #print(rps)
        eqs = rps.lemke_howson(0)
        next_state_value1 = rps[list(eqs)][0]
        
        ###########################################################
        act5 = random.randint(0,actions-1)
        act6 = random.randint(0,actions-1)
        
        

        p_s_new = np.random.random()
        p = 0
        s_new2 = -1
        while (p < p_s_new) and (s_new2 < (states - 1)):
            s_new2 = s_new2 + 1
            #print(a1,a2,s,s_new)
            p = p + P[act5][act6][s_new1][s_new2]
            
        r2 = R[act5][act6][s_new1]  
        
        rps = nash.Game(Q[s_new2,:,:])
        #print(rps)
        eqs = rps.lemke_howson(0)
        next_state_value2 = rps[list(eqs)][0]
        
        ##################################################################
        act7 = random.randint(0,actions-1)
        act8 = random.randint(0,actions-1)
        
        p_s_new = np.random.random()
        p = 0
        s_new3 = -1
        while (p < p_s_new) and (s_new3 < (states - 1)):
            s_new3 = s_new3 + 1
            #print(a1,a2,s,s_new)
            p = p + P[act7][act8][s_new2][s_new3]
            
        r3 = R[act7][act8][s_new2]  
        
        rps = nash.Game(Q[s_new3,:,:])
        #print(rps)
        eqs = rps.lemke_howson(0)
        next_state_value3 = rps[list(eqs)][0]
        
        ###############################################################
        # delta = (r + discount * next_state_value + discount * (1/ (np.log(n+2)) ) - Q[state, act1,act2])

        # delta = (r + discount * next_state_value + discount * (1/ (np.log(n+2)) ) * (r1 + discount * next_state_value1)- Q[state, act1,act2])
        delta = (r + discount * next_state_value + discount * (80/ (n+80) ) * (r1 + discount * next_state_value1) + discount**2 *  (10/ (n**2+10))*(r2 + discount * next_state_value2)+ discount**3 *  (10/ (n**2+10))*(r3 + discount * next_state_value3)- Q[state, act1,act2]) 
        #dQ = (1 / math.sqrt(np.sum(tot_count[act1][act2][state]))) * delta 
        #dQ = (1 / (np.sum(tot_count[act1][act2][state]))**0.501) * delta  
        dQ = (100/ (n+100)) * delta 
#         dQ = 0.9 * delta 
        Q[state, act1,act2] += dQ
        state = s_new3
        """
        sor_minimax_Q3 = np.zeros(states)
        for i in range(states):
            rps = nash.Game(Q[i,:,:])
            #print(rps)
            eqs = rps.lemke_howson(0)
    #         print(list(eqs))
            sor_minimax_Q3[i] =  rps[list(eqs)][0]
        mmql_output[avg][n] = np.linalg.norm(V - sor_minimax_Q3)  
        """
    sor_minimax_Q = np.zeros(states)
    for i in range(states):
        rps = nash.Game(Q[i,:,:])
        #print(rps)
        eqs = rps.lemke_howson(0)
#         print(list(eqs))
        sor_minimax_Q[i] =  rps[list(eqs)][0]
        
    endtime = time.time()
    store_avgs[avg][0] = np.linalg.norm(V - sor_minimax_Q)
    store_time[avg][0] = endtime - starttime 
#     print(np.linalg.norm(V - sor_minimax_Q))    
# y3 = mmql_output.mean(0)
# np.savetxt('y3.txt',y3, fmt='%d')
print(np.average(store_avgs),np.std(store_avgs))    
print('Four-step minimax Q-learning took {} seconds',np.mean(store_time))

print("------------------------------------------------")