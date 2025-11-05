import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, dense_size, output_size, dropout_p=0.2):
        super(LSTMModel, self).__init__()

        self.lstm1 = nn.LSTM(
            input_size,
            hidden_size_1,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout_p)


        self.lstm2 = nn.LSTM(
            hidden_size_1,
            hidden_size_2,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(dropout_p)

        self.fc1 = nn.Linear(hidden_size_2, dense_size)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(dense_size, output_size)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)

        out, _ = self.lstm2(out)
        out = self.dropout2(out[:, -1, :])

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out


class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, dense_size, output_size, dropout_p=0.2):
        super(BiLSTMModel, self).__init__()

        self.lstm1 = nn.LSTM(
            input_size,
            hidden_size_1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout1 = nn.Dropout(dropout_p)


        self.lstm2 = nn.LSTM(
            hidden_size_1 * 2,
            hidden_size_2,
            batch_first=True,
            bidirectional=True
        )
        self.dropout2 = nn.Dropout(dropout_p)

        self.fc1 = nn.Linear(hidden_size_2 * 2, dense_size)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(dense_size, output_size)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)

        out, _ = self.lstm2(out)
        out = self.dropout2(out[:, -1, :])

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out