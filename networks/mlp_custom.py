import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim=600, output_dim=2):
        super().__init__()

        # self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.input_fc = nn.Linear(input_dim, 300)
        self.hidden1_fc = nn.Linear(300, 150)
        # self.hidden2_fc = nn.Linear(250, 20)
        self.output_fc = nn.Linear(150, output_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = self.pool(x)
        # x = [batch size, data length]
        h_1 = self.input_fc(x)
        # h_1 = self.dropout(h_1)
        h_1 = self.relu(h_1)

        # h_1 = [batch size, 300]

        h_2 = self.hidden1_fc(h_1)
        # h_2 = self.dropout(h_2)
        h_2 = self.relu(h_2)

        # h_2 = [batch size, 150]

        # h_3 = self.hidden2_fc(h_2)
        # h_2 = self.dropout(h_2)
        # h_3 = self.relu(h_3)

        # h_3 = [batch size, 75]

        out = self.output_fc(h_2)

        return out # Criterion included Soft-max activation

if __name__ == '__main__':
    # data = [batch size, feature length]
    data = torch.rand((10, 1, 1024))
    print(data.shape)
    model = MLP(input_dim=1024, output_dim=2)
    output = model(data)
    print(output)
    print(output.shape)
