import torch
import torch.nn as nn
import math
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, input_size=20, hidden_layer_size=800, output_size=20):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear1 = nn.Linear(hidden_layer_size, math.ceil(1.25*hidden_layer_size))
        self.linear2 = nn.Linear(math.ceil(1.25*hidden_layer_size), output_size)

        self.dropout = nn.Dropout(p=0.2)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        drop_out = self.dropout(lstm_out)
        predictions = self.linear1(drop_out.view(len(input_seq), -1))
        #predictions = self.dropout(predictions)
        predictions = self.linear2(predictions)
        return predictions[-1]


class LSTM:
    def __init__(self, vocabSize, hidden_layer_size=800, lr=0.0001, tw=5, device=torch.device("cpu")):
        super().__init__()

        assert (tw != 0), "The training window has to be bigger than 0!"

        self.hidden_layer_size = hidden_layer_size

        self.vocabSize = vocabSize

        self.device = device

        self.model = LSTMModel(input_size=self.vocabSize, hidden_layer_size=self.hidden_layer_size, output_size=self.vocabSize).to(self.device)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.tw = tw

    # A class for creating the input tensors for the training with the defined training window
    def create_inout_sequences(self, input_data, tw=5):
        inout_seq = []
        for i in range(len(input_data)-tw):
            train_seq = input_data[i:i+tw]
            train_label = input_data[i+tw:i+tw+1]
            inout_seq.append((train_seq ,train_label))
        return inout_seq

    # The train main loop (teacher forcing)
    def train(self, tdata, epochs=700, verbose=False):
        self.trainingData = self.create_inout_sequences(torch.FloatTensor(tdata), self.tw)

        assert (len(tdata[0]) == self.vocabSize), "The number of features of the input tensor doesn't match the defined vocabSize!"

        if verbose:
            writer = SummaryWriter()

        for i in range(epochs):
            for seq, labels in self.trainingData:
                self.optimizer.zero_grad()
                self.model.hidden_cell = (torch.zeros(1, 1, self.model.hidden_layer_size).to(self.device),
                        torch.zeros(1, 1, self.model.hidden_layer_size).to(self.device))

                y_pred = self.model(seq.to(self.device))

                single_loss = self.loss_function(y_pred, labels.view(self.vocabSize).to(self.device))
                single_loss.backward()
                self.optimizer.step()

            if verbose:
                writer.add_scalar('Loss/train', single_loss.item(), i)

    # A class for making a prediction based on the recent n datapoints
    def predict(self, data, future=1):
        assert (len(data[0]) == self.vocabSize), "The number of features of the input tensor doesn't match the defined vocabSize!"
        inputList = data[-self.tw:,:]
        inputList = inputList.tolist()
        for i in range(future):
            seq = torch.FloatTensor(inputList[-self.tw:]).to(self.device)
            with torch.no_grad():
                self.model.hidden_cell = (torch.zeros(1, 1, self.model.hidden_layer_size).to(self.device),
                        torch.zeros(1, 1, self.model.hidden_layer_size).to(self.device))
                inputList.append(self.model(seq).cpu().numpy())
        return inputList[-future:]
