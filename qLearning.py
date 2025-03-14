import numpy as np
import matplotlib.pyplot as plt

class tic_tac_toe_agent():
    def __init__(self, q_table, boardstate):
        self.state = 's_0'
        self.q_table = {self.state : q_table} 
        self.boardstate = {self.state : boardstate}

        self.start_epsilon = 0.5
        self.decay_rate = 0.01
        self.min_epsilon = 0.01

        self.alfa = 1
        self.gamma = 0.1

    def get_state(self):
        '''
        returns current state
        Belongs to q-learning
        '''
        return self.state
    
    def update_state(self):
        '''
        Changes state.
        Belongs to q-learning
        '''
        last_state = list(self.boardstate.keys())[-1]
        nr = int(last_state.split('_')[1])
        last_state_plus_one = nr + 1
        new_state = f's_{last_state_plus_one}'
        self.state = new_state
    
    def set_state(self, state):
        '''
        Changes state.
        Belongs to q-learning
        '''
        self.state = state

    def get_valid_actions(self, q_table):
        '''
        state: current q table

        return: posible actions from state x
        
        returns what actions are possible from current state
        check in boardstate where the 

        Belongs to q-learning
        '''
        return np.argwhere(~np.isnan(q_table))
    
    def get_q_table(self):
        '''
        returns q table of current state
        Belongs to q-learning
        '''
        return self.q_table[self.state]
    
    def set_q_table(self, curr_q_table, action):
        '''
        adds a new state and corresponding actions to q table
        '''
        #self.q_table[self.state][action[0], action[1]] = np.nan
        next_q_table = curr_q_table.copy()
        next_q_table[action[0], action[1]] = np.nan
        return next_q_table
    
    def get_boardstate(self):
        return self.boardstate[self.state]

    def set_boardstate(self, curr_boardstate, action, player):
        #self.boardstate[self.state][action[0], action[1]] = player*1
        next_boardstate = curr_boardstate.copy()
        next_boardstate[action[0], action[1]] = player*1
        return next_boardstate

    def choose_action(self, current_q_table, episode):
        '''
        returns action. epsilon-greedy.
        Belongs to q-learning.
        '''
        r = np.random.rand()
        epsilon = max(self.start_epsilon * np.exp(-self.decay_rate*episode), self.min_epsilon)
        actions = self.get_valid_actions(current_q_table)
        #if len(actions) > 1:
        #    r_int = np.random.randint(0, len(actions))
        #    actions = actions[r_int]
        if actions is None or len(actions) == 0:
            return None
        if r < epsilon: # random
            r_int = np.random.randint(0, len(actions))
            return actions[r_int]
        elif r > epsilon: # greedy
            max_value = np.nanmax(current_q_table)
            pos = np.argwhere(current_q_table == max_value)
            if len(pos) > 1:
                r_int = np.random.randint(0, len(pos))
                pos = pos[r_int]
            else:
                pos = pos[0]
            return pos # maybe problem when max is 0


    def occured_previously(self, curr_q_table, curr_boardstate):
        '''
        checks if a boardstate has occured previously. If not
        q table is updated with a new state 
        '''
        
        for state, boardstate in self.boardstate.items(): # running through all board states that has occured
            if np.array_equal(boardstate, curr_boardstate): 
                return True, state
        return False, None  

    def update_q_value_function(self, state, previous_state, action, reward):
        '''
        Belongs to q-learning
        '''
        pos = np.argwhere(self.q_table[state] == np.nanmax(self.q_table[state]))
        if pos.size == 0:
            max_q = 0
        else:
            max_q = self.maxQ(pos, self.q_table[state])
        self.q_table[previous_state][action[0]][action[1]] += self.alfa*(reward + self.gamma*(max_q - self.q_table[previous_state][action[0]][action[1]]))

    def maxQ(self, pos, q_table):
        '''
        This function calculates tha maximal future reward togheter with update_q_value().
        Belongs to q-learning.
        '''
        if len(pos) > 1:
            r_int = np.random.randint(0, len(pos))
            pos = pos[r_int]
        else:
            pos = pos[0]
        return q_table[pos[0], pos[1]]
    

    def update_q_table_and_boardstate(self, action, curr_q_table, curr_boardstate, reward, player):
        '''
        input: new state and new boardstate
        output: a new addition to q table and boardstate
        update q-table part belongs to q-learning
        '''
        
        new_q_table = self.set_q_table(curr_q_table, action) # sets NaN.is copy
        new_boardstate = self.set_boardstate(curr_boardstate, action, player) # sets -1 or 1
        old_state = self.get_state()
        found, state = self.occured_previously(new_q_table, new_boardstate)

        if not found:
            self.update_state()
            self.q_table.setdefault(self.state, new_q_table)
            self.boardstate.setdefault(self.state, new_boardstate)
        else:
            self.set_state(state)
        
        self.update_q_value_function(self.state, old_state, action, reward)

        
        return new_q_table, new_boardstate

    def insert_new_state(self, new_q_table, new_boardstate):
        found, state = self.occured_previously(new_q_table, new_boardstate)
        if not found:
            self.update_state()
            self.q_table.setdefault(self.state, new_q_table)
            self.boardstate.setdefault(self.state, new_boardstate)
        else:
            self.set_state(state)


class Game_simulation():
    def __init__(self):
        self.player_x, self.player_o = self.initialize_players()
        self.goal = False
        self.episodes = 500

    def initialize_players(self):
        player_x = tic_tac_toe_agent(np.zeros((3,3), dtype=np.float64), np.zeros((3,3), dtype=np.float64) )
        player_o = tic_tac_toe_agent(np.zeros((3,3), dtype=np.float64), np.zeros((3,3), dtype=np.float64) )

        #state = player_x.get_state()
        #curr_q_table = player_x.get_q_table()#[state]
        #curr_boardstate = player_x.get_boardstate()#[state]

        #action = player_x.choose_action(curr_q_table, 1)
        #new_q_table, new_boardstate = player_x.update_q_table_and_boardstate(action, curr_q_table, curr_boardstate, reward=0, player=1)

        #player_o.q_table['s1'] = new_q_table 
        #layer_o.boardstate['s1'] = new_boardstate 
        return player_x, player_o

    def check_winner(self, matrix):
        reward = 1
        for row in matrix:
            if np.all(row == row[0]) and row[0] != 0:  # Ensure not all are zero (empty)
                self.goal = True
                winner = row[0]
                return reward, winner  # Return the winning value

        # Check columns
        for col in matrix.T:  # Transpose to check columns as rows
            if np.all(col == col[0]) and col[0] != 0:
                self.goal = True
                winner = col[0]
                return reward, winner

        # Check main diagonal
        if np.all(np.diag(matrix) == matrix[0, 0]) and matrix[0, 0] != 0:
            self.goal = True
            winner = matrix[0,0]
            return reward, winner

        # Check anti-diagonal (bottom-left to top-right)
        if np.all(np.diag(np.fliplr(matrix)) == matrix[0, 2]) and matrix[0, 2] != 0:
            self.goal = True
            winner = matrix[0,2]
            return reward, winner

        return 0, None  # No winner

    def simulation(self):
        winner = np.zeros(self.episodes)
        runs = np.zeros((10, len(winner)))
        '''
        Game overview:

        One player starts and chooses an action based or epsilon-greedy.
        the action updates the q table with a possibly new state
        the boardstate updates as well. This happens for both players. 
        Player x then chooses an action 
        '''
        for enum, r in enumerate(runs, axis=-1):
            for episode in range(self.episodes):
                print(f'{episode/self.episodes} %')
                #self.player_x.q_table = {'s_0': np.zeros((3,3), dtype=np.float64)}
                #self.player_o.q_table = {'s_0': np.zeros((3,3), dtype=np.float64)}
                self.player_x.set_state('s_0')
                self.player_o.set_state('s_0')
                starting_player = np.random.choice([-1, 1]) 
                while True:
                    if starting_player == -1:
                        state = self.player_o.get_state()

                        reward, w = self.check_winner(self.player_x.boardstate[state])
                        if w == 1:
                            self.player_x.update_q_value_function(state=self.player_x.get_state(), previous_state=old_state_x, action=action, reward=reward)
                            self.player_o.update_q_value_function(state=self.player_o.get_state(), previous_state=old_state_o, action=old_action_o, reward=-reward)
                            winner[episode] = w
                            break

                        curr_q_table = self.player_o.get_q_table()
                        curr_boardstate = self.player_o.get_boardstate()
                        if np.abs(np.sum(curr_boardstate)) > 1:
                            print(curr_boardstate)

                        action = self.player_o.choose_action(curr_q_table, episode)
                        
                        old_state_o = state
                        old_action_o = action
                        if action is None: # if there are no more actions and nobody has three in a row the game is over. 
                            self.player_x.update_q_value_function(state=self.player_x.get_state(), previous_state=old_state_x, action=old_action_x, reward=0.5)
                            winner[episode] = 0
                            break
                        new_q_table, new_boardstate = self.player_o.update_q_table_and_boardstate(action, curr_q_table, curr_boardstate, reward, player=-1)
                        self.player_x.insert_new_state(new_q_table, new_boardstate)
                        
                        
                        state = self.player_x.get_state()

                        reward, w = self.check_winner(self.player_x.boardstate[state])
                        if w == -1:
                            self.player_o.update_q_value_function(state=self.player_o.get_state(), previous_state=old_state_o, action=action, reward=reward)
                            self.player_x.update_q_value_function(state=self.player_x.get_state(), previous_state=old_state_x, action=old_action_x, reward=-reward)
                            winner[episode] = w
                            break
                            
                        
                        curr_q_table = self.player_x.get_q_table()#[state]
                        curr_boardstate = self.player_x.get_boardstate()#[state]
                        

                        action = self.player_x.choose_action(curr_q_table, episode)

                        old_state_x = state
                        old_action_x = action
                        if action is None: # if there are no more actions and nobody has three in a row the game is over. 
                            self.player_o.update_q_value_function(state=self.player_o.get_state(), previous_state=old_state_o, action=old_action_o, reward=0.5)
                            winner[episode] = 0
                            break
                        new_q_table, new_boardstate = self.player_x.update_q_table_and_boardstate(action, curr_q_table, curr_boardstate, reward, player=1)
                        self.player_o.insert_new_state(new_q_table, new_boardstate)
                    else:
                        state = self.player_x.get_state()

                        reward, w = self.check_winner(self.player_o.boardstate[state])
                        if w == -1:
                            self.player_o.update_q_value_function(state=self.player_o.get_state(), previous_state=old_state_o, action=action, reward=reward)
                            self.player_x.update_q_value_function(state=self.player_x.get_state(), previous_state=old_state_x, action=old_action_x, reward=-reward)
                            winner[episode] = w
                            break

                        curr_q_table = self.player_x.get_q_table()
                        curr_boardstate = self.player_x.get_boardstate()
                        if np.abs(np.sum(curr_boardstate)) > 1:
                            print(curr_boardstate)

                        action = self.player_x.choose_action(curr_q_table, episode)
                        
                        old_state_x = state
                        old_action_x = action
                        if action is None: # if there are no more actions and nobody has three in a row the game is over. 
                            self.player_o.update_q_value_function(state=self.player_o.get_state(), previous_state=old_state_o, action=old_action_o, reward=0.5)
                            winner[episode] = 0
                            break
                        new_q_table, new_boardstate = self.player_x.update_q_table_and_boardstate(action, curr_q_table, curr_boardstate, reward, player=1)
                        self.player_o.insert_new_state(new_q_table, new_boardstate)
                        
                        
                        state = self.player_o.get_state()

                        reward, w = self.check_winner(self.player_o.boardstate[state])
                        if w == 1:
                            self.player_x.update_q_value_function(state=self.player_x.get_state(), previous_state=old_state_x, action=action, reward=reward)
                            self.player_o.update_q_value_function(state=self.player_o.get_state(), previous_state=old_state_o, action=old_action_o, reward=-reward)
                            winner[episode] = w
                            break
                            
                        
                        curr_q_table = self.player_o.get_q_table()#[state]
                        curr_boardstate = self.player_o.get_boardstate()#[state]
                        

                        action = self.player_o.choose_action(curr_q_table, episode)

                        old_state_o = state
                        old_action_o = action
                        if action is None: # if there are no more actions and nobody has three in a row the game is over. 
                            self.player_x.update_q_value_function(state=self.player_x.get_state(), previous_state=old_state_x, action=old_action_x, reward=0.5)
                            winner[episode] = 0
                            break
                        new_q_table, new_boardstate = self.player_o.update_q_table_and_boardstate(action, curr_q_table, curr_boardstate, reward, player=-1)
                        self.player_x.insert_new_state(new_q_table, new_boardstate)
                    
                    
            runs[enum, :] = winner       
        x = np.arange(0,episode+1)
        #print(np.sum(np.where(winner == -1)))
        #print(np.sum(np.where(winner == 0)))
        #print(np.sum(np.where(winner == 1)))
        #plt.scatter(x, winner)

        plt.show()
                    
                
game = Game_simulation()
game.simulation()