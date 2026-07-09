import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

#----------------------------#
# Building pd to tensor      #
#----------------------------#

#The PandasDataSet class defined is a custom class that extends TensorDataset, 
# a dataset class provided by the PyTorch library.
#This class is designed to convert pandas DataFrame or Series objects into PyTorch tensors,
#which can then be used in a PyTorch model for training or inference.
class PandasDataSet(TensorDataset):

    '''
    This class lets you pass pandas objects directly into a PyTorch dataset.
    '''
    def __init__(self, *dataframes):
        
        tensors = (self._df_to_tensor(df) for df in dataframes) 

        #after converting pandas DataFrames into tensors, the class initializes itself as a normal PyTorch TensorDataset.
        super().__init__(*tensors)

    #This method checks if the input df is a pandas Series. If it is, it converts the Series to a DataFrame.
    #This is necessary because Series (1d) and DataFrame (2d) have different dimensionalities.
    #and for consistency, you might prefer everything as a DataFrame.

    #df.values extracts the numpy array from the DataFrame or Series.
    #torch.from_numpy(df.values).float() converts this numpy array into a PyTorch tensor
    #of floating-point numbers. The .float() method is called to ensure the tensor's datatype is float,
    #which is commonly used in neural network computations for higher precision.

    def _df_to_tensor(self, df):
        if isinstance(df, pd.Series):
            df = df.to_frame('temp') #The name does not matter much because column names are discarded when converting to NumPy/tensor.
        return torch.from_numpy(df.values).float()


#----------------------------#
# Building Classifier        #
#----------------------------#
class Classifier(nn.Module):

    '''
    When using nn.Sequential, you don't need to manually define the forward method in your neural network
    class. The sequential container automatically passes the output of each module
    as the input to the next, following the order in which they were added.
    This eliminates the need to manually route inputs and outputs through the layers.

    '''

    def __init__(self, n_features, n_hidden=32, p_dropout=0.2):
        super(Classifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.network(x)) #converts the final linear layer's output to a probability between 0 and 1


#----------------------------#
# Building Pretrain Classifier#
#----------------------------#
#Before the adversarial fairness game begins, you need a decent baseline model.

# For each epoch, we'll iterate over the batches returned by our `DataLoader`.
def pretrain_classifier(clf, data_loader, optimizer, criterion):
    for x, y, _ in data_loader:
        clf.zero_grad()
        p_y = clf(x) #define yhat
        loss = criterion(p_y, y) #calculates the error between the predicted probability p_y and the true label y using defined loss function (e.g. BCE or CrossEntropyLoss in the criterion arg)
        loss.backward() #calculate gradients
        optimizer.step() #update weights
    return clf

#----------------------------#
# Building Adversary        #
#----------------------------#
#look at the predictions made by the Classifier and try to guess the person's sensitive attributes (e.g., race or sex).

class Adversary(nn.Module):

    def __init__(self, n_sensitive, n_hidden=32):
        super(Adversary, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, n_hidden), #input is the probability of high income predicted by the classifier (yhat)
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_sensitive),
        )

    def forward(self, x):
        return torch.sigmoid(self.network(x))

#----------------------------#
# Building Pretrain Adversary#
#----------------------------#
# Training the adversary is pretty similar to how we trained the classifier.
# Note that we `.detach()` the predictions of the classifier from the graph.
# This SIGNALS PyTorch that we don't use the gradients of the classifier operations.
# during this pretraining step, only adv is trained to optimize the adversary, clf is frozen because of .detach()
# allowing PyTorch to free up some memory.

def pretrain_adversary(adv, clf, data_loader, optimizer, criterion, lambdas):
    '''
    The goal is to train adv, not clf.
    '''
    for x, _, z in data_loader:
        p_y = clf(x).detach() #get the yhat from the classifier and detach it from the graph
        #The adversary is trained on top of the classifier’s output, but the classifier itself is not changed.
        optimizer.zero_grad()
        p_z = adv(p_y)
        loss = (criterion(p_z, z) * lambdas).mean() #lambdas is a weighting factor. It controls how strongly each loss contributes.
        loss.backward()
        optimizer.step()
    return adv


#--------------------------#
# define train function    #
#--------------------------#

def fairness_train(clf, adv, data_loader,
                   clf_criterion, adv_criterion,
                   clf_optimizer, adv_optimizer, lambdas,
                   update_frequency=None):
    '''
    One epoch of the zero-sum fairness game between classifier and adversary.

    The adversary learns on the full data set and the classifier is given
    fewer updates, giving the adversary an edge in learning.

    update_frequency=None (default):
    the adversary trains on every batch, then the classifier is updated ONCE
    on the last batch of the epoch.

    update_frequency=k (LW's changed version) Instead of updating the classifier only once at the end, it updates the classifier every k batches.
    Update Frequency: The code handles a schedule where the Adversary gets to train on many batches for every 1 time the Classifier trains. 
    This ensures the Adversary stays strong, forcing the Classifier to work harder to be fair.
    '''

    if update_frequency is None:
        # Step 1: the adversary is trained on the full epoch,
        # independently of the classifier
        for x, y, z in data_loader:
            p_y = clf(x).detach()
            adv.zero_grad()
            p_z = adv(p_y)
            loss_adv = (adv_criterion(p_z, z) * lambdas).mean()
            loss_adv.backward()
            adv_optimizer.step()

        # Step 2: the classifier is updated on a single batch (the last one) of the dataloader,
        # with its original loss MINUS the weighted adversarial loss because lower loss means better prediction
        p_y = clf(x)
        clf_optimizer.zero_grad()
        #Predict y well, but make z hard to predict.
        loss_clf = clf_criterion(p_y, y) - (adv_criterion(adv(p_y), z) * lambdas).mean() #adjusted loss function
        loss_clf.backward()
        clf_optimizer.step()

    else:
        update_counter = 0 #Initializes a counter to track how many updates have been processed.
        for x, y, z in data_loader:
            # Always train the adversary first
            p_y = clf(x).detach()
            adv.zero_grad()
            p_z = adv(p_y)
            loss_adv = (adv_criterion(p_z, z) * lambdas).mean()
            loss_adv.backward()
            adv_optimizer.step()

            # Train the classifier conditionally
            #Scheduled Updates: Update the classifier every k iterations
            #to allow the adversary more training steps in between.
            update_counter += 1
            if update_counter % update_frequency == 0: #So the adversary trains every batch, but the classifier trains only every k batches.
                p_y = clf(x)
                clf_optimizer.zero_grad()
                loss_clf = clf_criterion(p_y, y) - (adv_criterion(adv(p_y), z) * lambdas).mean()
                loss_clf.backward()
                clf_optimizer.step()

    return clf, adv
