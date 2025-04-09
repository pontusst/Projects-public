import pandas as pd¨
import numpy as np


def read_data():
    df = pd.read_csv("C:/Users/pontu/Desktop/CHALMERS/PROJECTS_REPO/Project_ideas/change_this.csv")
    return df

def main():
    '''
    
    '''
    df = read_data()
    ratings = df['Ratings']
    scores = df['Scores']
    nr_of_samples = len(ratings)
    input_matrix = np.zeros((2, nr_of_samples))
    label = ratings
    input_matrix[:,0], input_matrix[:,1] = 1, scores

    ### training
    moment_matrix = np.matmul(input_matrix.T, input_matrix)
    inv_moment_matrix = np.invert(moment_matrix)
    beta = np.matmul(inv_moment_matrix,np.matmul(input_matrix.T,label))

    ### prediction
    new_x_matrix = np.ones((2, nr_of_samples))
    new_x_matrix[:,1] = 'par'
    pred = np.matmul(new_x_matrix.T, beta)