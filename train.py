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
    # データ　ローダー　を 準備しよう！
    # TODO: Build data loader.
    train_dataset = SurnameDataset(subset='train')
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataset = SurnameDataset(subset='val')
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True)

    # Generate a name for the run
    run_name = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    if not os.path.isdir('./log/'):
        os.mkdir('./log/')
    log_path = './log/{}/'.format(run_name)
    os.mkdir(log_path)
    writer = SummaryWriter(log_path)

    # モデル　と　オプティマイザ　は何でしょうか？
    # TODO: Build the network, loss and optimizer.
    net = SurnameLSTM()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1)

    # トレニング　１００エポック　を しましょう！
    # TODO: Train the model for 100 epochs.
    global_step = 0
    for i in range(100):
        for j, sample in enumerate(train_data_loader):
            # 勾配クリア　を　忘れないで〜
            # TODO: Remember to clear the gradient every time.
            net.zero_grad()

            # プリディクション　と　コストの計算　を　しましょう！
            # TODO: Prediction and loss calculation.
            output = net(sample['values'])
            loss = criterion(output, sample['label'])

            # 精度　を　計算しませんか？
            # TODO: Calculate accuracy for the whole batch.
            pred = np.argmax(output.detach().numpy(), axis=1)
            gt = np.argmax(sample['label'].numpy(), axis=1)
            accuracy = np.average(np.where(pred == gt, 1, 0))

            print('Epoch {}, Iteration {}, Loss = {:.3f}, Accuracy = {:.1f} %'.format(i + 1, j + 1, loss,
                                                                                      accuracy * 100))
            # 最後に、バックプロパゲーション　を　しましょう！
            # TODO: Finally, time for back-propagation.
            loss.backward()
            optimizer.step()

            # バリデーション　を　しませんか？
            # TODO: Do validation every 10 iterations.
            if global_step % 10 == 0:
                sample = iter(val_data_loader).__next__()

                output = net(sample['values'])
                loss = criterion(output, sample['label'])

                pred = np.argmax(output.detach().numpy(), axis=1)
                gt = np.argmax(sample['label'].numpy(), axis=1)
                accuracy = np.average(np.where(pred == gt, 1, 0))

                print('Validation Loss = {:.3f}, Accuracy = {:.1f} %'.format(loss, accuracy * 100))

            # TensorboardX で　ロッグ　を　しませんか？
            # TODO: Log the loss values and accuracies with TensorboardX.
            global_step += 1
            writer.add_scalar('loss', loss, global_step=global_step)
            writer.add_scalar('accuracy', accuracy, global_step=global_step)

    # モデル　は　まだ使うつもりですが。。。
    # TODO: Save log and weights.
    writer.close()
    print('Training log saved to {}'.format(log_path))
    state_dict = net.state_dict()
    torch.save(state_dict, 'model.pth')
    print('Model weights saved to ./model.pth')
