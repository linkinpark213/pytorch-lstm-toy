import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from net import SurnameLSTM
from data import SurnameDataset
from tensorboardX import SummaryWriter

if __name__ == '__main__':
    # データ　ローダー　を準備しよう！
    # TODO: Build data loader.
    train_dataset = None
    train_data_loader = None
    val_dataset = None
    val_data_loader = None

    # モデル　と　オプティマイザ　は何でしょうか？
    # TODO: Build the network, loss and optimizer.
    net = None
    loss = None
    optimizer = None

    # トレニング　１００エポック　を しよう！
    # TODO: Train the model for 100 epochs.
    for i in range(100):
        for j, sample in enumerate(train_data_loader):
            # 勾配クリア　を　忘れないで〜
            # TODO: Remember to clear the gradient every time.

            # プリディクション　と　コストの計算　を　しよう！
            # TODO: Prediction and loss calculation.

            # 精度　を　計算しない？
            # TODO: Calculate accuracy for the whole batch.

            # 最後に、バックプロパゲーション　を　しよう！
            # TODO: Finally, time for back-propagation.

            # バリデーション　を　しない？
            # TODO: Do validation every 10 iterations.

            # TensorboardX で　ロッグ　を　しない？
            # TODO: Log the loss values and accuracies with TensorboardX?


            pass

    # モデル　は　まだ使うつもりですが。。。
    # TODO: Save log and weights.
