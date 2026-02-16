import numpy as np
import matplotlib.pyplot as plt
from src.core.reward_table import Q
from src.core.board import Board
import random

class Agent():
    def __init__(self, q_table, boardstate, name):
        self.name = name

        self.previous_action = None
        self.previous_bs1 = (0,0,0,0,0,0,0,0,0)
        self.mark = None
        self.Q = Q(boardstate, q_table) # q-table is used to check for best action

        self.start_epsilon = 1.0
        self.decay_rate = 0.001
        self.min_epsilon = 0.01

        self.alfa = 0.1
        self.gamma = 0.99
    

    def get_valid_actions(self, board):
        return [i for i, v in enumerate(board) if v == 0]
    

    def set_boardstate(self, curr_boardstate, action, mark):
        return (mark if idx == action else i for idx, i in enumerate(curr_boardstate))

    def choose_action(self, current_board, curr_q, episode):
        '''
        returns action. epsilon-greedy.
        Belongs to q-learning.
        '''
        r = np.random.rand()
        epsilon = max(self.start_epsilon * np.exp(-self.decay_rate*episode), self.min_epsilon)
        v_actions = self.get_valid_actions(current_board)

        if v_actions is None or len(v_actions) == 0:
            return None
        if r < epsilon: # random
            r_int = np.random.randint(0, len(v_actions))
            return v_actions[r_int]
        
        elif r > epsilon: # greedy
            d = {a:b for a,b in curr_q.items() if a in v_actions}
            try:
                best_action = random.choice([a for a in v_actions if d[a] == max(d.values())])
            except:
                print('Choosing best action failed!')
            
            return best_action 


    def update_q_value_function(self, action, reward, pq, cq):
        '''
        Belongs to q-learning
        '''

        max_q = max(cq.values(), default=0) 
        
        pq[action] = (1-self.alfa)*pq[action] + self.alfa*(reward + self.gamma*max_q - pq[action])

        return cq
         
    
    def update_boardstate(self, action, curr_boardstate, mark):
        '''
        input: new state and new boardstate
        output: a new addition to q table and boardstate
        update q-table part belongs to q-learning
        '''
        
        #new_q_table = self.set_q_table(curr_q_table, action) # sets NaN.is copy
        new_boardstate = tuple(self.set_boardstate(curr_boardstate, action, mark))  # sets x or o

        return new_boardstate
    



                    
                
