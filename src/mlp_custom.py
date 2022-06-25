import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim=600, output_dim=2):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, 300)
        self.hidden_fc = nn.Linear(300, 150)
        self.output_fc = nn.Linear(150, output_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = [batch size, data length]
        h_1 = self.input_fc(x)
        # h_1 = self.dropout(h_1)
        h_1 = self.relu(h_1)

        # h_1 = [batch size, 300]

        h_2 = self.hidden_fc(h_1)
        # h_2 = self.dropout(h_2)
        h_2 = self.relu(h_2)

        # h_2 = [batch size, 150]

        out = self.output_fc(h_2)
        # m = nn.Sigmoid()
        # m = nn.Softmax()
        # y_pred = m(out)
        y_pred = out # Criterion included Soft-max activation

        # y_pred = [batch size, output dim]

        return y_pred

if __name__ == '__main__':
    # data = [batch size, feature length]
    data = torch.rand((10, 600))
    print(data.shape)
    model = MLP(input_dim=600, output_dim=2)
    output = model(data)
    print(output)
    print(output.shape)
