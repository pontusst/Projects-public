from src.agent import Agent
import numpy as np
from src.utils import check_winner, REG, R
from src.results import Results
from src.core.board import Board
from collections import deque
import matplotlib.pyplot as plt
import time


class Game_simulation():
    def __init__(self):
        
        self.goal = False
        self.episodes = 10
        self.games = 5000
        self.results = Results()

    def initialize_players(self):
        q1 = np.zeros(9, dtype=np.float64)
        b1 = (0,0,0,0,0,0,0,0,0)

        q2 = np.zeros(9, dtype=np.float64)
        b2 = (0,0,0,0,0,0,0,0,0)
        player_x = Agent(q1, b1, name='player_X')
        player_o = Agent(q2, b2, name='player_O')
        
        player_x.mark = 1
        player_o.mark = -1
 
        return player_x, player_o

    def apply_switch(self, player_1, player_2):
        p1 = player_2
        p2 = player_1

        return p1, p2

    def simulation(self):
        start = time.time()
        runs = np.zeros((self.episodes, self.games))
        player_x, player_o = self.initialize_players()
        PLAYERS = {'player_X' : player_x,
                   'player_O': player_o}
        '''
        Game overview:

        One player starts and chooses an action based on epsilon-greedy.
        the action updates the q table with a new state
        the boardstate updates as well. This happens for both players. 
        Player x then chooses an action. 
        '''
        
        for enum, r in enumerate(runs):
            self.results.winner = []
            for game in range(self.games):
                p = game/self.games
                per = np.round(p, 2)*100
                print(f'{per} %')
                choise = np.random.choice([-1, 1]) 
                player_1 = PLAYERS[REG[choise]]
                player_2 = PLAYERS[REG[-choise]]
                
                bs1 = player_1.Q.get_keys()[0]
                count = 0
                player_1.previous_action = None
                player_2.previous_action = None
                
                while True:
                        if count != 0:
                            # what does player 1 see?
                            bs1 = player_1.Q.curr_b

                        # check if someone has won!
                        reward, w = check_winner(bs1)

                        try:
                            w = REG[w]
                            if w == PLAYERS[w].name:
                                n = PLAYERS[w].previous_bs1
                                
                                PLAYERS[w].update_q_value_function(PLAYERS[w].previous_action, reward, PLAYERS[w].Q.curr_Q[n], {})
                                if PLAYERS[w].name == 'player_X':
                                    nn = player_o.previous_bs1
                                    player_o.update_q_value_function(player_o.previous_action, -10, player_o.Q.curr_Q[nn], {})
                                elif PLAYERS[w].name == 'player_O':
                                    nn = player_x.previous_bs1
                                    player_x.update_q_value_function(player_x.previous_action, -10, player_x.Q.curr_Q[nn], {})

                                self.results.winner.append(PLAYERS[w].mark)
                                break
                        except:
                            pass

                        curr_q_table = player_1.Q.curr_Q[bs1]
                        curr_boardstate = bs1


                        # current player chooses an action based on q table
                        action = player_1.choose_action(curr_boardstate, curr_q_table, game)

                        
                        if action is None: # if there are no more actions and nobody has three in a row the game is over.
                            player_2.update_q_value_function(player_2.previous_action, 0, player_2.Q.curr_Q[player_2.previous_bs1], {})
                            player_1.update_q_value_function(player_1.previous_action, 0, player_1.Q.curr_Q[player_1.previous_bs1], {})
                            self.results.winner.append(None)
                            break
                        
                        # update board state and q table for curr player. 
                        
                        new_b = player_1.update_boardstate(action, curr_boardstate, mark=player_1.mark)
                        player_1.Q.add(new_b)
                        player_2.Q.add(new_b)


                        if player_1.previous_action is not None:
                            prev_b = player_1.previous_bs1
                            prev_action = player_1.previous_action
                            player_1.update_q_value_function(prev_action, reward, player_1.Q.curr_Q[prev_b], player_1.Q.curr_Q[bs1])

                            
                        player_1.previous_action = action
                        player_1.previous_bs1 = bs1
                        
                        player_1, player_2 = self.apply_switch(player_1, player_2)

                        assert player_1.Q.curr_b == player_2.Q.curr_b
                        count += 1
                    
            
            end = time.time()
            print(f'running time : {end-start} s')
            runs[enum, :] = self.results.winner    
            one = np.cumsum(np.array(self.results.winner) == 1)
            n_one = np.cumsum(np.array(self.results.winner) == -1)
            none = np.cumsum(np.array(self.results.winner) == None)
            r = np.arange(0,self.games)

            plt.figure()

            #plt.subplot(1, 3, 1)
            plt.scatter(r, one, label='one', s=1)
            #plt.legend()

            #plt.subplot(1,3, 2)
            plt.scatter(r, n_one ,label='n_one', s=1)
            #plt.legend()

            #plt.subplot(1,3, 3)
            plt.scatter(r, none, label='none', s=1)
            plt.legend()
            plt.show()

        #x = np.arange(0,+1)