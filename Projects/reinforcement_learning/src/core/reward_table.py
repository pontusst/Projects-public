import numpy as np

class Q():
    def __init__(self, bstate, q_values):
        self.actions = (0,1,2,3,4,5,6,7,8)
        self.q_list = {self.actions[i] : q_values[i] for i in range(len(self.actions))}

        self.curr_Q = {bstate : self.q_list}
        self.curr_b = bstate

    def add(self, new_b):
        q_values = (0,0,0,0,0,0,0,0,0)
        q_list = {self.actions[i] : q_values[i] for i in range(len(self.actions))}
        b = self.curr_Q.setdefault(new_b, q_list)
        if isinstance(b, tuple):
            self.curr_b = b
        else:
            self.curr_b = new_b
    
    def get_keys(self):
        l = [b for b in self.curr_Q.keys()]
        return l
        

    
        

