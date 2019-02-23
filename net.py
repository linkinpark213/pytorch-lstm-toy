import torch
import torch.nn as nn


class SurnameLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # 深層ネットワーク　を　作りましょう！
        # TODO: Build the whole network.

        # まず　ASCIIコード　を　ベクトル　に　しましょう！
        # TODO: 1. Map the ASCII codes to embeddings.

        # LSTM層？
        # TODO: 2. Add the LSTM layer.

        # Dense層？
        # TODO: 3. Add some dense layers and activation functions.

        # Dropout層？
        # TODO: 4. Add a dropout layer.

        # Softmax　クラシフィケーション層？
        # TODO: 5. Add a softmax layer for classification.

    def forward(self, input: torch.Tensor):
        # フォワードプロパゲーション　を　しましょう！
        # TODO: Implement the forward-propagation.
        
        pass
