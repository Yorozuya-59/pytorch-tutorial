from tqdm import tqdm

from icecream import ic

import torch

from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor, Lambda

from torch.utils.data import DataLoader

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from torcheval.metrics import MulticlassAccuracy

from mylib.initialize import init_settings, set_seeds


init_settings(verbose=True)     # 環境中での初期化処理を実行する自作関数．PyTorch のチュートリアルとは関係しないため，よしなに書き換えてください
set_seeds(seed=42)     # 再現性の確保のために乱数シードを設定する自作関数．


'''
定数の宣言（Python では標準で変更不可能な定数の宣言はできないため，慣習的にすべて大文字の変数名を使用する）
'''
NUM_TARGET = 10    # 今回利用するデータセットである FashionMNIST は 10 クラス分類問題である
LEARNING_RATE = 1e-3
EPOCHS = 100
LR_STEP_SIZE = 10
LR_GAMMA = 0.5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU が利用可能であれば GPU を利用する


data_root_dir = '/home/workdir/datasets'     # データセットの保存先ディレクトリ．複数回読み込む場合に，ダウンロードすることなく，保存したデータセットを利用することができる

'''
Dataset の取得
今回は FashionMNIST データセットを利用する
https://github.com/zalandoresearch/fashion-mnist
自前のデータを利用したい場合， Dataset クラスを継承して独自のクラスを実装することで利用可能となる
今回は PyTorch のチュートリアルで，簡単のために提供されているものを利用する
'''
train_data = FashionMNIST(
    root=data_root_dir,
    train=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(NUM_TARGET, dtype=torch.float).scatter(0, torch.tensor(y), value=1)),
    download=True,
)
test_data = FashionMNIST(
    root=data_root_dir,
    train=False,
    transform=ToTensor(),
    download=True,
)

'''
DataLoader の作成
DataLoader は Dataset からデータをバッチで取り出すためのイテレータを提供する
'''
train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=64,
    shuffle=True,
)
test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=64,
    shuffle=False,
)


'''
ニューラルネットワークの定義
nn.Module クラスを継承して定義し，このクラスの定義に必要最低限な関数は __init__ と forward の2つである．
'''
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        '''
        ネットワークの要素を定義する関数
        ここで定義した要素を forward 関数で呼び出し，順伝播の計算を定義する
        '''
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=3,
            stride=1,
            padding=0
        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            padding=0
        )

        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            padding=1
        )

        self.flatten = nn.Flatten(
            start_dim=1
        )

        self.fc = nn.Linear(
            in_features=(4 ** 2) * 16,
            out_features=1024
        )

        self.fc_out = nn.Linear(
            in_features=1024,
            out_features=NUM_TARGET
        )
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        '''
        順伝播の計算を定義する関数
        ネットワークの要素を定義した __init__ 関数で定義した要素を呼び出して，計算の順序を定義する
        '''
        h = self.conv1(X)
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.flatten(h)
        
        h = self.relu(self.fc(h))
        y = self.fc_out(h)

        return self.softmax(y)


cnn = ConvolutionalNeuralNetwork()
ic(cnn)


'''
学習に必要な要素の定義
1. 損失関数：今回は多クラス分類問題であるため，交差エントロピー誤差を利用する
2. 最適化手法：今回は Adam を利用する
3. （任意）学習率スケジューラ：今回は StepLR を利用する
'''
loss_fn = nn.CrossEntropyLoss().to(DEVICE)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
criterion = MulticlassAccuracy().to(DEVICE)
model = cnn.to(DEVICE)


'''
学習ループ
1. 学習フェーズ
2. 評価フェーズ
3. （任意）学習率スケジューラの更新
'''
for epoch in tqdm(range(EPOCHS), desc='Epoch'):
    current_lr = scheduler.get_last_lr()[0]
    ic(epoch, current_lr)
    
    '''
    学習フェーズ
    1. ミニバッチ単位でデータを取得
    2. モデルによる推論（順伝播）
    3. 損失関数の計算
    4. 勾配の計算（逆伝播）
    5. パラメータの更新
    '''
    model.train()     # 学習モードに設定（Dropout や BatchNorm を利用する場合に必要）
    for inputs, targets in tqdm(train_dataloader, desc=f'Epoch {epoch+1} / {EPOCHS}', leave=False):     # 1. ミニバッチ単位でデータを取得
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        predicts = model(inputs)     # 2. モデルによる推論（順伝播）
        loss = loss_fn(predicts, targets)     # 3. 損失関数の計算

        optimizer.zero_grad()     # optimizer の勾配を初期化
        loss.backward()     # 4. 勾配の計算（逆伝播）
        optimizer.step()     # 5. パラメータの更新
        
        predicts = predicts.argmax(dim=1)
        targets = targets.argmax(dim=1)
        criterion.update(predicts, targets)     # 推論結果の記録

    acc = criterion.compute()     # スコアの計算（今回は multiclass_accuracy を利用）
    ic('Train', epoch, acc)
    criterion.reset()     # スコア計算のための状態をリセット
    
    
    '''
    評価フェーズ
    1. ミニバッチ単位でデータを取得
    2. モデルによる推論（順伝播）
    3. スコアの計算
    また，評価フェーズではモデルのパラメータを更新しないため，torch.no_grad() を利用して，勾配計算を無効化することで，メモリ使用量の削減と計算高速化を図る
    '''
    model.eval()     # 評価モードに設定（Dropout や BatchNorm を利用する場合に必要）
    with torch.no_grad():
        for inputs, targets in tqdm(test_dataloader, desc=f'Epoch {epoch+1} / {EPOCHS}', leave=False):     # 1. ミニバッチ単位でデータを取得
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            predicts = model(inputs)     # 2. モデルによる推論（順伝播）
            
            predicts = predicts.argmax(dim=1)
            # targets = targets.argmax(dim=1)
            criterion.update(predicts, targets)     # 推論結果の記録
    
    acc = criterion.compute()     # 3. スコアの計算
    ic('Evaluate', epoch, acc)
    criterion.reset()     # スコア計算のための状態をリセット

    scheduler.step()     # 学習率スケジューラの更新

