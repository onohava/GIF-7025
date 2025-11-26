# model implementation inspired by paper: An attention-based LSTM network for large earthquake prediction (doi: 10.1016/j.soildyn.2022.107663)

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention_weights = nn.Linear(hidden_size, 1, bias=True)

    def forward(self, lstm_output):
        scores = self.attention_weights(lstm_output)
        probs = F.softmax(scores, dim=1)

        weighted = lstm_output * probs
        context_vector = torch.sum(weighted, dim=1)
        return context_vector, probs


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, dense_size_1, dense_size_2, output_size):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            batch_first=True,
        )

        self.attention = Attention(hidden_size)
        self.leaky_relu = nn.LeakyReLU(0.01)

        self.fc1 = nn.Linear(hidden_size, dense_size_1)
        self.fc2 = nn.Linear(dense_size_1, dense_size_2)
        self.fc3 = nn.Linear(dense_size_2, output_size)


    def forward(self, x):
        out, _ = self.lstm(x)
        out, attention_weights = self.attention(out)
        out = self.leaky_relu(self.fc1(out))
        out = self.leaky_relu(self.fc2(out))

        return self.fc3(out)