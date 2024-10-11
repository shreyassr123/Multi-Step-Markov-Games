#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 16:49:05 2023

@author: Shreyas S R FINAL
"""


import numpy as np
import nashpy as nash
from copy import deepcopy
import math
import random
import time
import matplotlib.pyplot as plt


np.random.seed(100)
random.seed(100)
 
states = 50
actions = 5
discount = 0.6
max_iterations = 1000 #100000

total_avgs = 50

mmql_output = np.zeros((total_avgs,max_iterations))
ql_output = np.zeros((total_avgs,max_iterations))
sorql_output = np.zeros((total_avgs,max_iterations))
sorql_opt_output = np.zeros((total_avgs,max_iterations))

P = np.zeros((actions,actions,states,states))
R = np.random.random((actions,actions,states))

for a1 in range(actions):
    for a2 in range(actions):
        for s in range(states):
            P[a1,a2,s] = np.random.random(states)
            P[a1][a2][s][s] = states #check this. 
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
        dQ = (1 / (np.sum(tot_count[act1][act2][state]))**0.501) * delta  
#         dQ = 0.9 * delta 
        Q[state, act1,act2] += dQ
        state = s_new
        """
        minimax_Q1 = np.zeros(states)
        for i in range(states):
            rps = nash.Game(Q[i,:,:])
            #print(rps)
            eqs = rps.lemke_howson(0)
    #         print(list(eqs))
            minimax_Q1[i] =  rps[list(eqs)][0]
        ql_output[avg][n] = np.linalg.norm(V - minimax_Q1)    
        """
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
    #print(store_time[avg][0])
#     print(np.linalg.norm(V - sor_minimax_Q)) 
"""   
y1 = ql_output.mean(0)

np.savetxt('y1.txt',y1, fmt='%d')
"""
print(np.average(store_avgs),np.std(store_avgs))    
print('Standard Minimax Q-learning took {} seconds', np.mean(store_time))
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
        dQ = (1 / (np.sum(tot_count[act1][act2][state]))**0.501) * delta  
#         dQ = 0.9 * delta 
        Q[state, act1,act2] += dQ
        state = s_new1
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
    
    store_avgs[avg][0] = np.linalg.norm(V - sor_minimax_Q)
    
    endtime = time.time()
    store_time[avg][0] = endtime - starttime 
#     print(np.linalg.norm(V - sor_minimax_Q))    
"""
y3 = mmql_output.mean(0)
np.savetxt('y3.txt',y3, fmt='%d')
"""
print(np.average(store_avgs),np.std(store_avgs))    
print('Two-step minimax Q-learning took {} seconds',np.mean(store_time))

print("------------------------------------------------")




# SOR Q-learning 

store_avgs = np.zeros((total_avgs,1))
store_time = np.zeros((total_avgs,1))
P_update_time = max_iterations/10
store_w = np.zeros((max_iterations,total_avgs))
for avg in range(total_avgs):
    starttime = time.time()
    Q = np.random.rand(states,actions,actions)
    state = np.random.randint(0, states)
    

    tot_count = np.zeros((actions,actions,states,states))

    w = 1
    
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

#         if n > P_update_time:  #Doing it after initial 1000 iterations
        new_w = 1/(1-discount)
        for a1 in range(actions):
            for a2 in range(actions):
                for s in range(states):
                    if np.sum(tot_count[a1][a2][s][s]) > 0:
                        temp = 1/(1 - (discount*(tot_count[a1][a2][s][s]/np.sum(tot_count[a1][a2][s]))))
                        #print(temp)
                        if new_w > temp:
                            new_w = temp
                            
        step_size = 1 / math.sqrt(n+1)
        w = (1-step_size)*w+step_size*new_w


        store_w[n][avg] = w

        rps = nash.Game(Q[state,:,:])
        #print(rps)
        eqs = rps.lemke_howson(0)
        current_state_value = rps[list(eqs)][0]


        rps = nash.Game(Q[s_new,:,:])
        #print(rps)
        eqs = rps.lemke_howson(0)
        next_state_value = rps[list(eqs)][0]

        delta = r + discount * next_state_value - Q[state, act1,act2]
        delta = w *(r + discount* next_state_value) + (1-w)*current_state_value - Q[state, act1,act2]
        #dQ = (1 / math.sqrt(np.sum(tot_count[act1][act2][state]))) * delta
        dQ = (1 / (np.sum(tot_count[act1][act2][state]))**0.501) * delta  
        #         dQ = 0.9 * delta 
        Q[state, act1,act2] += dQ
        state = s_new
        """
        sor_minimax_Q2 = np.zeros(states)
        for i in range(states):
            rps = nash.Game(Q[i,:,:])
            #print(rps)
            eqs = rps.lemke_howson(0)
    #         print(list(eqs))
            sor_minimax_Q2[i] =  rps[list(eqs)][0]
        sorql_output[avg][n] = np.linalg.norm(V - sor_minimax_Q2) 
        """
    sor_minimax_Q = np.zeros(states)
    for i in range(states):
        rps = nash.Game(Q[i,:,:])
        #print(rps)
        eqs = rps.lemke_howson(0)
#         print(list(eqs))
        sor_minimax_Q[i] =  rps[list(eqs)][0]
    
    store_avgs[avg][0] = np.linalg.norm(V - sor_minimax_Q)
    endtime = time.time()
    store_time[avg][0] = endtime - starttime 
    
    # print(np.linalg.norm(V - sor_minimax_Q))    
"""    
y2 = sorql_output.mean(0)
np.savetxt('y2.txt',y2, fmt='%d')
"""
print(np.average(store_avgs),np.std(store_avgs))     
print('Model-free SOR Q-learning took {} seconds',np.mean(store_time))
print("------------------------------------------------")



#Generalised Optimal minimax Q-Learning



store_avgs = np.zeros((total_avgs,1))

P_update_time = max_iterations/10
store_w = np.zeros((max_iterations,total_avgs))
store_time = np.zeros((total_avgs,1))





for avg in range(total_avgs):
    starttime = time.time()
    w = 100
            
    for a1 in range(actions):
        for a2 in range(actions):
            for s in range(states):
                temp = 1/(1 - (discount*P[a1][a2][s][s]))
                if w > temp:
                    w = temp
                    
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

        store_w[n][avg] = w

        rps = nash.Game(Q[state,:,:])
        #print(rps)
        eqs = rps.lemke_howson(0)
        current_state_value = rps[list(eqs)][0]


        rps = nash.Game(Q[s_new,:,:])
        #print(rps)
        eqs = rps.lemke_howson(0)
        next_state_value = rps[list(eqs)][0]

        delta = r + discount * next_state_value - Q[state, act1,act2]
        delta = w *(r + discount* next_state_value) + (1-w)*current_state_value - Q[state, act1,act2]
        # dQ = (1 / math.sqrt(np.sum(tot_count[act1][act2][state]))) * delta 
        dQ = (1 / (np.sum(tot_count[act1][act2][state]))**0.501) * delta  
#         dQ = 0.9 * delta 
        Q[state, act1,act2] += dQ
        state = s_new

    sor_minimax_Q = np.zeros(states)
    for i in range(states):
        rps = nash.Game(Q[i,:,:])
        #print(rps)
        eqs = rps.lemke_howson(0)
#         print(list(eqs))
        sor_minimax_Q[i] =  rps[list(eqs)][0]
    
    store_avgs[avg][0] = np.linalg.norm(V - sor_minimax_Q)
    endtime = time.time()
    store_time[avg][0] = endtime - starttime 
#     print(np.linalg.norm(V - sor_minimax_Q))    

print(np.average(store_avgs),np.std(store_avgs))     

print('Generalised Optimal minimax Q-Learning took {} seconds',np.mean(store_time))

print('------------------------------------------------------------')

"""
figsize = 8, 4
figure, ax = plt.subplots(figsize=figsize)

x=range(max_iterations)

plt.axhline(y=0, color='grey', linestyle='dotted')
# plt.axhline(y=0.33, color='y')
# plt.axhline(y=-0.95, color='b')
lines=plt.plot(x,y1,x,y2,x,y3)
# lines=plt.plot(x,mean1,x,mean2,x,mean5,x,mean3)
# l1,l2,l3,l4=lines
l1, l2, l3=lines
# plt.setp(lines, linestyle='...')
plt.setp(l1, linewidth=1, color='r', label='MQL')
plt.setp(l2, linewidth=1, color='y', label='SORQL')
plt.setp(l3, linewidth=1, color='b', label='MMQL')

# plt.setp(l4, linewidth=1, color='g', label='TSQL2')

handles, labels = plt.gca().get_legend_handles_labels()

legend = plt.legend(loc='right', bbox_to_anchor=(1, 0.9),
      ncol=1, fancybox=True, shadow=True,prop={'size': 10})
legend.get_title().set_fontname('DejaVu Sans')
legend.get_title().set_fontweight('black')

# plt.tick_params(labelsize=20)
# labels = ax.get_xticklabels() + ax.get_yticklabels()
# [label.set_fontname('DejaVu Sans') for label in labels]

font2 = {'family': 'DejaVu Sans',
         'weight': 'black',
         'size': 10,
         }
plt.xlabel('Number of Iterations', font2)
plt.ylabel('Average Error', font2)
# plt.tight_layout()
# plt.show()

plt.savefig('shreyas_final.pdf', dpi=600, bbox_inches='tight')
plt.close()
"""