# Sarcini:
# - Scrierea și comentarea codului.
#  - Implimentarea clasei Data set.  
#  - Implimentarea rețelei neuronale.
#  - Implimentarea ciclului de învățarea.
#  - Prezența grficului  - Learning Curve pe acuratete și eroare.
#  - Reantrenarea modelului cu cel mai bun rezultat dupa learning curve.


#Important imports
from numpy.core.fromnumeric import argmax
import torch
from torch import nn as nn
from torch.utils.data import Dataset
from torch import optim 
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from torch.nn import functional as F

#Choosing devide to be gpu if have one, else cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  


#Creaating DataSet class with which we will transform our tabelar data into tensors
class DataSet(Dataset):

    def __init__(self,file_name):  
        #transformed csv file into tensor
        self.dataset = torch.tensor(pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)),file_name)).values)

    def __len__(self):
        #return lenght of dataset
        return len(self.dataset)

    def __getitem__(self, i):
        #returns X and Y data by index
        return self.dataset[i,:-1], self.dataset[i,-1]            ## mb add transformation that normalizes



class Net(nn.Module):

    def __init__(self, n_features):
        super().__init__()
        #Setting number of initial features, which will be equal to number of features in dataset
        self.n_features = n_features
        
        #self.norm = nn.LayerNorm(n_features)
        #Setting some layers for our architecture
        self.fc1 = nn.Linear(self.n_features,16) # (input, ouput)
        self.fc2 = nn.Linear(16,32) # (input, ouput)
        self.fc3 = nn.Linear(32,1)  # (input, ouput)           

    def forward(self,x):
        #Running forward cycle of our network
        out = F.silu(self.fc1(x))
        out = F.silu(self.fc2(out))
        out = F.sigmoid(self.fc3(out))
        
        #returning ouput
        return out

# Training loop for our network
def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, val_loader, l2=0.001, print_plot=True):

    #Initializing two lists to save loses to plot them later
    train_loses = []
    val_loses = []

    #Going thru every epoch
    for epoch in range(1,n_epochs+1):

        #Setting loss to 0 at the begginng of every epoch
        train_loss = 0.0
        val_loss = 0.0

        #Going thru example, thru every batch, in our case, thru all data at once
        for example, labels in train_loader:     

            #Translating calculations to gpu if is available
            example = example.to(DEVICE)
            labels = labels.to(DEVICE)

            # ensuring equal number of dimensions for labels and examples
            labels  = labels.unsqueeze(1)

            # running our data thru our data - forward
            train_output = model(example.float())
            # Getting loss of our network right now
            train_loss += loss_fn(train_output, labels.float())

            
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            train_loss = train_loss + l2 *l2_norm

            # Zeroing the gradient to not stack it from other iterations
            optimizer.zero_grad()
            #Runing backward part of the neural network, getting gradiets
            train_loss.backward()
            #Updating our paramters
            optimizer.step()

        #Running over validation data
        for example, labels in val_loader:

            #Translating calculations to gpu if is available
            example = example.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # ensuring equal number of dimensions for labels and examples
            labels  = labels.unsqueeze(1)

            #Forward
            val_output = model(example.float())
            #Loss
            val_loss += loss_fn(val_output, labels.float())


        #Print results for epochs
        if epoch == 1 or epoch % 5 == 0:
            print('Epoch {0}, Training loss - {1}, Validation loss {2}'.format(epoch,train_loss, val_loss))

        # Append losses to lists
        train_loses.append(train_loss)
        val_loses.append(val_loss)

    #If set to True, print graph of train and validation loss
    if print_plot:

        #Setting x-ticks
        epochs = range(1,n_epochs+1)

        #Ploting both curves, train and val 
        plt.plot(epochs, train_loses, 'g', label='Training loss')
        plt.plot(epochs, val_loses, 'b', label='validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
 
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

#Creating Dataset from csv file
data = DataSet('heart.csv')

#Saving nr of examples and calculating number of validation samples
n_samples = len(data)
n_val = int(0.2*n_samples)

#Spliting into train and validation
train_set, val_set = torch.utils.data.random_split(data,  [n_samples-n_val, n_val])


# running train and validation sets thru dataloader that helps with parallelizing the data loading process with automatic batching
train_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set,batch_size=len(val_set), shuffle=True)


#Setting some hyperparameters and parameters
learning_rate = 1e-2

#Initializing model with nr of features from input
model = Net(len(data[0][0])).to(DEVICE)

#Optimizer and loss funtion
optimizer = optim.Adam(model.parameters(),lr=learning_rate)
loss_fn = nn.BCELoss()                                                     

#Running training loop on the data with set parameters
training_loop(
    n_epochs=500,
    optimizer=optimizer,
    model = model,
    loss_fn = loss_fn,
    print_plot=True,
    train_loader=train_loader,
    val_loader = val_loader
)


#Renewing our model to run it for more efficient number of epochs


# for layer in model.children():
#    if hasattr(layer, 'reset_parameters'):                                                           ######### idk
#        layer.reset_parameters()

model = Net(len(data[0][0])).to(DEVICE)
optimizer = optim.Adam(model.parameters(),lr=learning_rate)
 

training_loop(
    n_epochs=100,
    optimizer=optimizer,
    model = model,
    loss_fn = loss_fn,
    print_plot=True,
    train_loader=train_loader,
    val_loader = val_loader
)




# print(val_set[:][0].float().to(DEVICE))
# from sklearn.metrics import accuracy_score
# pred_test = model(val_set[:][0].float().to(DEVICE))
# preds_y = pred_test.argmax().to(DEVICE)
# accuracy_score(val_set[:][1].float().to(DEVICE), preds_y)