from loguru import logger
import numpy as np
import pandas as pd
import seaborn as sns
import pytorch_lightning as pl
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
        return chart
    
    @staticmethod
    def calc_valid_loss(net, criterion, valid_loader, batch_size, use_gpu=False):
        """Run a model with valid_loader and calculate model loss.
        Call `net.train()` again after this function to resume training mode."""
        
        net.eval()
        val_losses = []
        for inputs, labels in valid_loader:
            if(use_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()

            output = net(inputs)
            val_loss = criterion(output, labels)

            val_losses.append(val_loss.item())
        return val_losses

    @classmethod
    def train_net(cls, net, criterion, optimizer, train_loader, valid_loader, batch_size, epochs, use_gpu=False, print_every=100, clip=10):
        logger.info(f"Training a model: epochs={epochs} use_gpu={use_gpu} clip={clip}")
        train_stat_dict = {
            "epoch":[],
            "step":[],
            "loss":[],
            "val_loss":[]
        }
        counter = 0

        # move model to GPU, if available
        if(use_gpu):
            net.cuda()

        net.train()
        # train for some number of epochs
        for e in range(epochs):
            train_losses = []
            # batch loop
            for inputs, labels in train_loader:
                counter += 1

                if(use_gpu):
                    inputs, labels = inputs.cuda(), labels.cuda()

                # zero accumulated gradients
                net.zero_grad()

                # get the output from the model
                output = net(inputs)
                
                # calculate the loss and perform backprop
                loss = criterion(output, labels)
                loss.backward()
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                nn.utils.clip_grad_norm_(net.parameters(), clip)
                optimizer.step()
                
                train_losses.append(loss.item())
                
                # Show loss stats every "print_every" batch
                if counter % print_every == 0:
                    val_losses = cls.calc_valid_loss(net, criterion, valid_loader, batch_size, use_gpu)
                    net.train()
                    logger.debug("epoch:{}/{} step={} train_loss={:.6f} val_loss={:.6f}".format(e+1, epochs, counter, np.mean(train_losses), np.mean(val_losses)))
            
            # End of epoch
            val_losses = cls.calc_valid_loss(net, criterion, valid_loader, batch_size, use_gpu)
            net.train()
            train_stat_dict["epoch"].append(e+1)
            train_stat_dict["step"].append(counter)
            train_stat_dict["loss"].append(np.mean(train_losses))
            train_stat_dict["val_loss"].append(np.mean(val_losses))

        return train_stat_dict

    @staticmethod
    def test_net(net, criterion, test_loader, batch_size, use_gpu=False):
        logger.info(f"Testing a model: use_gpu={use_gpu}")
        # Get test data loss and accuracy
        
        test_losses = [] # track loss
        preds = []
        true_labels = []

        if(use_gpu):
            net.cuda()
        
        net.eval()
        # iterate over test data
        for inputs, labels in test_loader:
            if(use_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()

            # get predicted outputs
            output = net(inputs)
            
            # calculate loss
            test_loss = criterion(output, labels)

            # get the predicted class by the highest probabilty
            top_p, pred = output.topk(1, dim=1)
            
            if use_gpu:
                # Move back GPU's memory to CPU's memory to compute score
                labels = labels.cpu().tolist()
                pred = pred.cpu().flatten().tolist()
            else:
                labels = labels.tolist()
                pred = pred.flatten().tolist()
            
            test_losses.append(test_loss.item())
            preds.extend(pred)
            true_labels.extend(labels)
            


        # -- stats! -- ##
        # avg test loss
        avg_loss = np.mean(test_losses)
        logger.info("Test loss: {:.3f}".format(avg_loss))
        
        return avg_loss, true_labels, preds
    
    @staticmethod
    def save_model_weight(net, model_path):
        logger.info(f"Saving model weight: path={model_path}")
        torch.save(net.state_dict(), model_path)
    
    @staticmethod
    def load_model_weight(net, model_path):
        logger.info(f"Loading model weight: path={model_path}")
        net.load_state_dict(torch.load(model_path))
        return net

class HarLSTM(pl.LightningModule):
    
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

        self.criterion = nn.CrossEntropyLoss()
        
    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")
    
    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "valid")
    
    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "test")
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, _ = self._prepare_batch(batch)
        return self(x)
    
    def _common_step(self, batch, batch_idx, stage: str):
        x, labels = self._prepare_batch(batch)
        loss = self.criterion(self(x), labels)
        self.log(f"{stage}_loss", loss, on_step=True)
        return loss
    
    def _prepare_batch(self, batch):
        # Ignore label
        x, labels = batch
        # Input shape should be (batch_size, seq_length, input_size)
        return x.view(x.size(0), -1, self.input_size), labels
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        out = self.dropout(lstm_out)
        # getting the last time step output
        out = lstm_out[:, -1, :] 
        
        # you may need to use contiguous to reshape the output
        out = out.contiguous().view(-1, self.n_hidden)
        out = self.fc(out)
        out = self.softmax(out)
    
        return out
    