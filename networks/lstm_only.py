import torch
# Preliminaries


import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class LSTM_Only(nn.Module):

    def __init__(self, img_dim=3, embedding_dim=2048, hidden_dim=128, output_dim=2, n_layers=2, bidirectional=True, dropout=0.2):
        super(LSTM_Only, self).__init__()

        # self.vgg16 = nn.Sequential(
        #     # conv1
        #     nn.Conv2d(img_dim, 64, (3, 3), padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, (3, 3), padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2),
        #
        #     # conv2
        #     nn.Conv2d(64, 128, (3, 3), padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 128, (3, 3), padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2),
        #
        #     # conv3
        #     nn.Conv2d(128, 256, (3, 3), padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, (3, 3), padding=1),
        #     nn.ReLU(),
        #     # nn.Conv2d(256, 256, (3, 3), padding=1),
        #     # nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2),
        #
        #     # conv4
        #     nn.Conv2d(256, 512, (3, 3), padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 512, (3, 3), padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 512, (3, 3), padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2),
        #
        #     # conv5
        #     nn.Conv2d(512, 512, (3, 3), padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 512, (3, 3), padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 512, (3, 3), padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2)
        # )
        #
        # self.fc_vgg = nn.Sequential(
        #     # FC layer
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(4096, 1000),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(1000, 256)
        # )

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
        # self.fc = nn.Linear(hidden_dim * 4, output_dim)
        # Prediction activation function
        # self.sigmoid = nn.Sigmoid

        self.combination = nn.Sequential(
            # nn.Linear(512, 256),
            # nn.Dropout(0.6),
            nn.Linear(256, 128),
            nn.Dropout(0.6),
            nn.Linear(128, 64),
            nn.Dropout(0.6),
            nn.Linear(64, output_dim),
        )

    def forward(self, x1, text_seq):

        # x1 = self.vgg16(x1)
        # x1 = x1.reshape(x1.size()[0], -1)
        # x1 = self.fc_vgg(x1)

        packed_output, (hidden_state, cell_state) = self.lstm(text_seq)
        # Concatenating the final forward and backward hidden states
        x2 = torch.cat((hidden_state[-2, :, :], hidden_state[-1, :, :]), dim=1)

        # Concatenating VGG and LSTM
        # x = torch.cat((x1, x2), 1)
        #
        dense_outputs = self.combination(x2)

        return dense_outputs

if __name__ == '__main__':
    # data = [batch size, feature length]
    img = torch.rand((10, 3, 224, 224))
    data = torch.rand((10, 1, 2048))
    print(data.shape)
    model = LSTM_Only(img_dim=3, embedding_dim=2048, output_dim=2)
    output = model(img, data)
    print(output)
    print(output.shape)
