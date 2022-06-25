import torch
import torch.nn as nn
# from networks.bert.bert import BERT
from transformers import BertModel, BertTokenizer, AutoModel
import torch.nn.functional as F

class vgg_bert(nn.Module):
    def __init__(self, in_ch=3, out_ch=2):
        super().__init__()

        self.vgg16 = nn.Sequential(
            # conv1
            nn.Conv2d(in_ch, 64, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            # conv2
            nn.Conv2d(64, 128, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            # conv3
            nn.Conv2d(128, 256, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding=1),
            nn.ReLU(),
            # nn.Conv2d(256, 256, (3, 3), padding=1),
            # nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            # conv4
            nn.Conv2d(256, 512, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            # conv5
            nn.Conv2d(512, 512, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.fc_vgg = nn.Sequential(
            # FC layer
            nn.Linear(512 * 7 * 7, 4096),
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

        self.bert = nn.Sequential(
            #BertModel.from_pretrained('bert-base-cased'),
            AutoModel.from_pretrained("GroNLP/bert-base-dutch-cased"),
            nn.Dropout(0.5),
            nn.Linear(768, 256),
            nn.ReLU()
        )

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
        x1 = self.vgg16(x1)
        x1 = x1.reshape(x1.size()[0], -1)
        x1 = self.fc_vgg(x1)

        for b_layer in self.bert:
            if isinstance(b_layer, BertModel):
                _, x2 = b_layer(input_ids=x2, attention_mask=mask, return_dict=False)
            else:
                x2 = b_layer(x2)

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

    model = vgg_bert()
    output = model(img, id, mask)
    print(output)
    print(output.shape)
    a = F.softmax(output, dim=1)
    print(a)
