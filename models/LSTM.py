import torch



class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
