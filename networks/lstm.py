import torch
# Preliminaries


import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from torchtext import data



class LSTMNet(nn.Module):

    def __init__(self, embedding_dim=600, hidden_dim=64, output_dim=2, n_layers=2, bidirectional=True, dropout=0.2):
        super(LSTMNet, self).__init__()

        # Embedding layer converts integer sequences to vector sequences
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer process the vector sequences
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=True
                            )

        # Dense layer to predict
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        # Prediction activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, text_lengths=None):
        # embedded = self.embedding(text)
        # embedded = text

        # Thanks to packing, LSTM don't see padding tokens
        # and this makes our model better
        # packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True)
        packed_embedded = text

        packed_output, (hidden_state, cell_state) = self.lstm(packed_embedded)

        # Concatenating the final forward and backward hidden states
        hidden = torch.cat((hidden_state[-2, :, :], hidden_state[-1, :, :]), dim=1)

        dense_outputs = self.fc(hidden)

        # Final activation function
        # outputs = self.sigmoid(dense_outputs)

        return dense_outputs

if __name__ == '__main__':
    # data = [batch size, feature length]
    inputs = [torch.randn(1, 3) for _ in range(5)]
    for i in inputs:
        print(i)
        i = i.view(1, 1, -1)
        print(i.shape)
    data = torch.rand((10, 1, 256))
    print(data.shape)
    model = LSTMNet(embedding_dim=256, output_dim=2)
    output = model(data)
    print(output)
    print(output.shape)
