import numpy as np
import torch 
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from  torch.utils.data import DataLoader, TensorDataset
import time
import matplotlib.pyplot as plt

class RNN_config():
    input_dim = 1 # nr of expected features of input 
    hidden_dim = 32 # nr of features in the hidden state
    num_layers = 2 # nr of recurrent layers, setting it to two equates to stacking two LSTM on top of eachother. 
    output_dim = 1
    num_epochs = 100
    

conf = RNN_config()

df = pd.read_csv("C:/Users/pontu/Desktop/CHALMERS/PROJECTS_REPO/Project_ideas/AABA_2006-01-01_to_2018-01-01.csv")
### Use average true range ###

### Normalizing data ### (v1) Using MinMaxScaler
price = df[['Close']]
scaler = MinMaxScaler(feature_range=(-1, 1))
price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1,1))

### Normalizing data (v2) using yesterdays close.
price_close= df[['Close']]
price_high = df[['High']]
price_low = df[['Low']]

# Compute first atr
atr_range = 14
for inx, i in enumerate(price_close):
    
atr_1 = [i for i in max()]

atr = "r" # average true range 



### Splitting data 
def split_data(stock, lookback):
    data_raw = stock.to_numpy() # convert to numpy array
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
    
    data = np.array(data)
    test_set_size = int(np.round(0.2*data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]
    
    return [x_train, x_test, y_train, y_test]
lookback = 20 # choose sequence length
data = split_data(price, lookback)

x_train = torch.from_numpy(data[0]).type(torch.Tensor)
x_val = torch.from_numpy(data[1]).type(torch.Tensor)
y_train_lstm = torch.from_numpy(data[2]).type(torch.Tensor)
y_val_lstm = torch.from_numpy(data[3]).type(torch.Tensor)

#win_data = torch.from_numpy(data).type(torch.Tensor)

#window_data = TensorDataset(win_data)
#n_train = int(0.9 * len(window_data))
#n_val = len(window_data) - n_train
#training_data, validation_data = torch.utils.data.random_split(window_data, (n_train, n_val))
#training_loader = DataLoader(training_data, batch_size=16, shuffle=True)
#validation_loader = DataLoader(validation_data, batch_size=16, shuffle=True)

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out



model = LSTM(input_dim=conf.input_dim, hidden_dim=conf.hidden_dim, output_dim=conf.output_dim, num_layers=conf.num_layers)
criterion = torch.nn.MSELoss(reduction='mean')
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    
hist = np.zeros(conf.num_epochs)
start_time = time.time()
lstm = []
tr_loss = np.zeros(conf.num_epochs)
val_loss = np.zeros(conf.num_epochs)
#x_train = training_loader
for t in range(conf.num_epochs):
    y_train_pred = model(x_train)
    y_val_pred = model(x_val)
    t_loss = criterion(y_train_pred, y_train_lstm)
    tr_loss[t] = t_loss
    val_loss[t] = criterion(y_val_pred, y_val_lstm)
    print("Epoch ", t, "MSE: ", tr_loss[t].item())
    hist[t] = tr_loss[t].item()
    optimiser.zero_grad()
    t_loss.backward()
    optimiser.step()
    
training_time = time.time()-start_time
print("Training time: {}".format(training_time))
# to numpy 
x_train = x_train.detach().numpy()
x_val = x_val.detach().numpy()
tr = np.arange(0,conf.num_epochs)
vl = np.arange(0,conf.num_epochs)

plt.plot(tr, tr_loss, label='training loss')
plt.plot(vl, val_loss, label='Validation loss')
plt.legend()
plt.show()
index_list = df.index.tolist()
plt.plot(index_list[:len(x_train)], price_close[:len(x_train)])
#plt.plot(index_list[len(x_train):],  ) # plot model predictions
plt.show()


"""
Where i left off:
reading about nomalization of data to not introduce look forward bias. 
Using average true range as a data instead of closing price. 
"""

