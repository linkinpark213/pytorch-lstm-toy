import torch
import torch.nn as nn


class SurnameLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # 深層ネットワーク　を　作りましょう！
        # TODO: Build the whole network.

        # まず　ASCIIコード　を　ベクトル　に　しましょう！
        # TODO: 1. Map the ASCII codes to embeddings.
        self.embedding = nn.Embedding(128, 128)

        # LSTM層？
        # TODO: 2. Add the LSTM layer.
        self.lstm = nn.LSTM(128, 32, 1)

        # Dense層？
        # TODO: 3. Add some dense layers and activation functions.
        self.fc1 = nn.Linear(32, 256)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(256, 3)

        # Dropout層？
        # TODO: 4. Add a dropout layer.
        self.dropout = nn.Dropout(p=0.2)

        # Softmax　クラシフィケーション層？
        # TODO: 5. Add a softmax layer for classification.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input: torch.Tensor):
        # フォワードプロパゲーション　を　しましょう！
        # TODO: Implement the forward-propagation.
        embedded_input = self.embedding(input.transpose(0, 1).type(torch.LongTensor))
        output, (ht, ct) = self.lstm(embedded_input)
        out = self.fc1(ht[0])
        out = self.sigmoid(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.softmax(out)
        return out
