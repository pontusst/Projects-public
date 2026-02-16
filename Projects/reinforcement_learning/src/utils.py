import numpy as np

def check_winner(matrixx):
        reward = 10
        matrix = np.array(matrixx).reshape(3, 3)

        for row in matrix:
            if np.all(row == row[0]) and row[0] != 0:  # Ensure not all are zero (empty)
                #self.goal = True
                winner = row[0]
                return reward, winner  # Return the winning value

        # Check columns
        for col in matrix.T:  # Transpose to check columns as rows
            if np.all(col == col[0]) and col[0] != 0:
                #self.goal = True
                winner = col[0]
                return reward, winner

        # Check main diagonal
        if np.all(np.diag(matrix) == matrix[0, 0]) and matrix[0, 0] != 0:
            #self.goal = True
            winner = matrix[0,0]
            return reward, winner

        # Check anti-diagonal (bottom-left to top-right)
        if np.all(np.diag(np.fliplr(matrix)) == matrix[0, 2]) and matrix[0, 2] != 0:
            #self.goal = True
            winner = matrix[0,2]
            return reward, winner

        return 0, None  # No winner


REG = {1 : 'player_X',
            -1 : 'player_O'}
R = {0:'x', 
     1:'o'}