from loguru import logger
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

class ModelUtils():
    
    @staticmethod
    def plot_loss_chart(train_stat_dict):
        train_stat_df = pd.DataFrame(data=train_stat_dict)
        train_stat_df.drop(["epoch"], axis=1, inplace=True)

        train_stat_df = train_stat_df.melt(id_vars=["step"])
        train_stat_df.rename({"variable":"type"}, axis=1, inplace=True)

        chart = sns.lineplot(data=train_stat_df, x="step", y="value", hue="type")
        display(chart)
    
    @staticmethod
    def train_net(net, criterion, optimizer, train_loader, valid_loader, batch_size, epochs, train_on_gpu=False, print_every=100, clip=10):
        logger.info(f"Training NN: epochs={epochs} train_on_gpu={train_on_gpu} clip={clip}")
        train_stat_dict = {
            "epoch":[],
            "step":[],
            "loss":[],
            "val_loss":[]
        }
        counter = 0

        # move model to GPU, if available
        if(train_on_gpu):
            net.cuda()

        net.train()
        # train for some number of epochs
        for e in range(epochs):
            # initialize hidden state
            h = net.init_hidden(batch_size, train_on_gpu)

            # batch loop
            for inputs, labels in train_loader:
                counter += 1

                if(train_on_gpu):
                    inputs, labels = inputs.cuda(), labels.cuda()

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                h = tuple([each.data for each in h])

                # zero accumulated gradients
                net.zero_grad()

                # get the output from the model
                output, h = net(inputs, h)
                
                # calculate the loss and perform backprop
                loss = criterion(output, labels)
                loss.backward()
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                nn.utils.clip_grad_norm_(net.parameters(), clip)
                optimizer.step()

                # loss stats
                if counter % print_every == 0:
                    # Get validation loss
                    val_h = net.init_hidden(batch_size, train_on_gpu)
                    val_losses = []
                    net.eval()
                    for inputs, labels in valid_loader:

                        # Creating new variables for the hidden state, otherwise
                        # we'd backprop through the entire training history
                        val_h = tuple([each.data for each in val_h])

                        if(train_on_gpu):
                            inputs, labels = inputs.cuda(), labels.cuda()

                        output, val_h = net(inputs, val_h)
                        val_loss = criterion(output, labels)

                        val_losses.append(val_loss.item())

                    net.train()
                    logger.debug("epoch:{}/{} step={} train_loss={:.6f} val_loss={:.6f}".format(e+1, epochs, counter, loss.item(), np.mean(val_losses)))

                    train_stat_dict["epoch"].append(e+1)
                    train_stat_dict["step"].append(counter)
                    train_stat_dict["loss"].append(loss.item())
                    train_stat_dict["val_loss"].append(np.mean(val_losses))

        return train_stat_dict

    @staticmethod
    def test_net(net, criterion, test_loader, batch_size, train_on_gpu=False):
        logger.info(f"Testing NN: train_on_gpu={train_on_gpu}")
        # Get test data loss and accuracy

        test_losses = [] # track loss
        accuracy_list = []
        precisions = []
        recalls = []

        # init hidden state
        h = net.init_hidden(batch_size, train_on_gpu)

        net.eval()
        # iterate over test data
        for inputs, labels in test_loader:

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            if(train_on_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()

            # get predicted outputs
            output, h = net(inputs, h)
            
            # calculate loss
            test_loss = criterion(output, labels)
            test_losses.append(test_loss.item())

            # get the predicted class by the highest probabilty
            top_p, pred = output.topk(1, dim=1)
            
            
            if train_on_gpu:
                # Move back to CPU-memory to compute score
                labels = labels.cpu().tolist()
                pred = pred.cpu().flatten().tolist()
            else:
                pred = pred.flatten().tolist()
            
            acc = accuracy_score(labels, pred)
            prec = precision_score(labels, pred, average="micro")
            rec = recall_score(labels, pred, average="micro")
            
            accuracy_list.append(acc)
            precisions.append(prec)
            recalls.append(rec)


        # -- stats! -- ##
        # avg test loss
        avg_loss = np.mean(test_losses)
        logger.info("Test loss: {:.3f}".format(avg_loss))
        
        test_acc = np.mean(accuracy_list)
        logger.info("Test accuracy: {:.6f}".format(test_acc))
        logger.info("Test precision: {:.6f}".format(np.mean(precisions)))
        logger.info("Test recall: {:.6f}".format(np.mean(recalls)))
        
        
        return avg_loss, test_acc

class HarLSTM(nn.Module):
    
    def __init__(self, input_size, output_size, n_hidden=256, n_layers=2,
                               drop_prob=0.5, lr=0.001):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        
        self.lstm = nn.LSTM(input_size, n_hidden, n_layers, 
                            dropout=drop_prob, batch_first=True)
        
        self.dropout = nn.Dropout(p=drop_prob)
        
        self.fc = nn.Linear(n_hidden, self.output_size)
        
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x, hidden):
        ''' Forward pass through the network. 
            These inputs are x, and the hidden/cell state `hidden`. '''
                
        batch_size = x.size(0)
        
        # get RNN outputs
        lstm_out, hidden = self.lstm(x, hidden)
        
        out = self.dropout(lstm_out)
        # getting the last time step output
        out = lstm_out[:, -1, :] 
        
        # you may need to use contiguous to reshape the output
        out = out.contiguous().view(-1, self.n_hidden)
        out = self.fc(out)
        out = self.softmax(out)
        
        # reshape to be batch_size first
#         out = out.view(batch_size, -1)
#         out = out[:, -1] # get last batch of labels
        return out, hidden
    
    
    def init_hidden(self, batch_size, train_on_gpu=False):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        
        return hidden