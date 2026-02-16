import numpy as np

class Board():
    def __init__(self, board):
        self.curr_board = board
        
    def add_board(self, new_board):
        for idx in range(0, len(self.curr_board), 9): # running through all board states that has occured
            if np.array_equal(self.curr_board[idx], new_board): 
                return idx
            
        self.curr_board = np.vstack((self.curr_board, new_board))
        return idx+1
            