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
actionsA = 4
actionsB = 6
discount = 0.5
max_iterations = 160

total_avgs = 1

TOTALMGs = 100
mmql_finalerror = np.zeros(TOTALMGs)
ql_finalerror = np.zeros(TOTALMGs)
store_time = np.zeros(TOTALMGs)
store_time1 = np.zeros(TOTALMGs)
from scipy.optimize import linprog


def find_minimax_value(game_matrix):
    """
    Find the min-max value of a given game matrix using linear programming.

    Parameters:
    - game_matrix (np.ndarray): The payoff matrix for the game.

    Returns:
    - float: The min-max value of the game.
    """
    actionsA, actionsB = game_matrix.shape

    # Objective function coefficients (negative for minimization)
    c = np.zeros(actionsA + 1)
    c[-1] = -1  # This is the value we want to minimize

    # Inequality constraints (A_ub * x <= b_ub)
    A_ub = np.hstack([-game_matrix.T, np.ones((actionsB, 1))])
    b_ub = np.zeros(actionsB)

    # Equality constraints (A_eq * x = b_eq)
    A_eq = np.ones((1, actionsA + 1))
    A_eq[0, -1] = 0  # The sum of probabilities for player 1 should be 1
    b_eq = np.array([1])

    # Bounds for the variables
    bounds = [(0, 1)] * actionsA + [(None, None)]

    # Solve the linear program using the 'highs' method
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if res.success:
        return -res.fun  # The min-max value is the negative of the result
    else:
        raise ValueError("Linear programming failed to find a solution.")


for totalmgs in range(TOTALMGs):
    mmql_output = np.zeros((total_avgs,max_iterations))
    ql_output = np.zeros((total_avgs,max_iterations))
    
    
    P = np.zeros((actionsA,actionsB,states,states))
    R = np.random.random((actionsA,actionsB,states))
    
    for a1 in range(actionsA):
        for a2 in range(actionsB):
            for s in range(states):
                P[a1,a2,s] = np.random.random(states)
                # P[a1][a2][s][s] = states #check this. 
                P[a1,a2,s] = P[a1,a2,s] / P[a1,a2,s].sum()
                
    #Value Iteration Starts
    if np.min(P) == 0:
        print("Error")
        break
    print(np.min(P))
    V = np.zeros(states) #Initial value
    while True:
        Q = np.zeros((actionsA,actionsB,states))
        for a1 in range(actionsA):
            for a2 in range(actionsB):
                Q[a1,a2] = R[a1,a2] + discount * P[a1,a2].dot(V)
    
        v_prev = deepcopy(V)
        for s in range(states):
            Q_s = Q[:,:,s]
            V[s] = find_minimax_value(Q_s)
    
        #print(v_prev)
        #print(V)
        #print(v_prev)
        #print(np.linalg.norm(V-v_prev))
    
        if np.linalg.norm(V-v_prev) < 0.000001:
            break
    
    # Standard minimax Q-learning
    
    
    store_avgs = np.zeros((total_avgs,1))
    
    P_update_time = max_iterations/10
    starttime = time.time()
    for avg in range(total_avgs):
        
    
        Q = np.random.rand(states,actionsA,actionsB)
        state = np.random.randint(0, states)
        
        tot_count = np.zeros((actionsA,actionsB,states,states))
    
        
    
        for n in range(max_iterations):
    
            if (n % 100) == 0:
                state = np.random.randint(0, states)
    
            act1 = random.randint(0,actionsA-1)
            act2 = random.randint(0,actionsB-1)
    
    
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
    
    
            
    
    
            # Find next state value using linear programming
            next_state_value = find_minimax_value(Q[s_new, :, :])
    
            delta = r + discount * next_state_value - Q[state, act1,act2]
            #delta = w *(r + discount* next_state_value) + (1-w)*current_state_value - Q[state, act1,act2]
            #dQ = (1 / math.sqrt(np.sum(tot_count[act1][act2][state]))) * delta  
            #dQ = (1 / (np.sum(tot_count[act1][act2][state]))**0.5) * delta  
            dQ = (100 /( n+100)) * delta  
    #         dQ = 0.9 * delta 
            Q[state, act1,act2] += dQ
            state = s_new
            
        minimax_Q = np.zeros(states)
        for i in range(states):
            # Find minimax value for each state using linear programming
            minimax_Q[i] = find_minimax_value(Q[i, :, :])
        
        store_avgs[avg][0] = np.linalg.norm(V - minimax_Q)
        
    endtime = time.time()
    store_time[totalmgs] = endtime - starttime 
    ql_finalerror[totalmgs] = np.average(store_avgs)
    
    #print(np.average(store_avgs),np.std(store_avgs))    
    
    
    
    
    # Two-step minimax Q-learning
    
    
    
    store_avgs = np.zeros((total_avgs,1))
    
    P_update_time = max_iterations/10
    starttime = time.time()
    for avg in range(total_avgs):
        
        Q = np.random.rand(states,actionsA,actionsB)
        state = np.random.randint(0, states)
        
        tot_count = np.zeros((actionsA,actionsB,states,states))
    
        
    
        for n in range(max_iterations):
    
            if (n % 100) == 0:
                state = np.random.randint(0, states)
    
            act1 = random.randint(0,actionsA-1)
            act2 = random.randint(0,actionsB-1)
    
    
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
    
    
    
    
    
            # Find next state value using linear programming
            next_state_value = find_minimax_value(Q[s_new, :, :])
            
            act3 = random.randint(0,actionsA-1)
            act4 = random.randint(0,actionsB-1)
    
            p_s_new = np.random.random()
            p = 0
            s_new1 = -1
            while (p < p_s_new) and (s_new1 < (states - 1)):
                s_new1 = s_new1 + 1
                #print(a1,a2,s,s_new)
                p = p + P[act3][act4][s_new][s_new1]
                
            r1 = R[act3][act4][s_new]  
            
            next_state_value1 = find_minimax_value(Q[s_new1, :, :])
            
            
            # delta = (r + discount * next_state_value + discount * (1/ (np.log(n+2)) ) - Q[state, act1,act2])
    
            # delta = (r + discount * next_state_value + discount * (1/ (np.log(n+2)) ) * (r1 + discount * next_state_value1)- Q[state, act1,act2])
            delta = (r + discount * next_state_value + discount * (1000/ (n+1000) ) * (r1 + discount * next_state_value1)- Q[state, act1,act2]) 
            #dQ = (1 / math.sqrt(np.sum(tot_count[act1][act2][state]))) * delta 
            #dQ = (1 / (np.sum(tot_count[act1][act2][state]))**0.5) * delta  
    #         dQ = 0.9 * delta 
            dQ = (100 /( n+100)) * delta  
            Q[state, act1,act2] += dQ
            state = s_new1
            
        sor_minimax_Q = np.zeros(states)
        for i in range(states):
            sor_minimax_Q[i] = find_minimax_value(Q[i, :, :])
            
        
        store_avgs[avg][0] = np.linalg.norm(V - sor_minimax_Q)
        
    
    endtime = time.time()
    store_time1[totalmgs] = endtime - starttime 
    mmql_finalerror[totalmgs] = np.average(store_avgs)
    



print('Statndard QL Error is ', np.mean(ql_finalerror))
print('Standard Minimax Q-learning took {} seconds',np.average(store_time))
print("------------------------------------------------")

print("Multi-step QL Error is",np.mean(mmql_finalerror))
print('Multi-step minimax Q-learning took {} seconds',np.mean(store_time1))
