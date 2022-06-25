import torch
import timm
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AutoModel
import torch.nn.functional as F


class Effb0_Bert(nn.Module):
    def __init__(self, out_ch=2, language='En', pre_trained=True):
        super().__init__()

        self.effb0 = timm.create_model('efficientnet_b0', features_only=True, pretrained=pre_trained)

        self.fc_effb0 = nn.Sequential(
            # FC layer
            nn.Linear(320 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(4096, 1000),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1000, 256)
        )

        if 'En' in language:
            self.bert = nn.Sequential(
                BertModel.from_pretrained('bert-base-cased'),
                # AutoModel.from_pretrained("GroNLP/bert-base-dutch-cased"),
                nn.Dropout(0.5),
                nn.Linear(768, 256),
                nn.ReLU()
            )
        else:
            self.bert = nn.Sequential(
                # BertModel.from_pretrained('bert-base-cased'),
                AutoModel.from_pretrained("GroNLP/bert-base-dutch-cased"),
                nn.Dropout(0.5),
                # nn.Linear(768, 256),
                # nn.ReLU()
            )
        self.lstm = nn.LSTM(768, 256, batch_first=True,bidirectional=True)

        self.combination = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.6),
            nn.Linear(256, 128),
            nn.Dropout(0.6),
            nn.Linear(128, 64),
            nn.Dropout(0.6),
            nn.Linear(64, out_ch),
            # nn.Sigmoid()
        )

    def forward(self, x1, x2, mask):
        """
        :param x1: image input
        :param x2: Bert ID
        :param mask: Bert Mask
        :return:
        """
        _, _, _, _, x1 = self.effb0(x1)
        x1 = x1.reshape(x1.size()[0], -1)
        x1 = self.fc_effb0(x1)

        for b_layer in self.bert:
            if isinstance(b_layer, BertModel):
                _, x2 = b_layer(input_ids=x2, attention_mask=mask, return_dict=False)
            else:
                x2 = b_layer(x2)
        x2, (h,c) = self.lstm(x2) ## extract the 1st token's embeddings
        x2 = torch.cat((x2[:,-1, :256],x2[:,0, 256:]),dim=-1)
        x = torch.cat((x1, x2), 1)
        x = self.combination(x)

        return x

if __name__ == '__main__':

    # img = [batch size, channel, width, height]
    img = torch.rand((1, 3, 224, 224))

    # id, mask = [batch size, token length]
    # id = torch.randint(1, 256, (1, 256))
    # mask = torch.randint(0, 1, (1, 256))

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    example_text = 'This is test code with Bert tokenizer'
    bert_input = tokenizer(example_text, padding='max_length', max_length=256,
                           truncation=True, return_tensors="pt")
    id = bert_input['input_ids']
    mask = bert_input['attention_mask']

    model = Effb0_Bert(out_ch=2, language='En')
    output = model(img, id, mask)
    print(output)
    print(output.shape)
    a = F.softmax(output, dim=1)
    print(a)