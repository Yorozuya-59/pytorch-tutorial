# 機械学習向けテンプレートリポジトリ


# フォルダ構成
Github上のフォルダ構成は以下の通り．
```bash: tree
.
├── README.md
├── docker-compose.yml
└── environment
    ├── .config
    │   └── ***
    └── Dockerfile.mlenv
```

`docker compose`実行後のフォルダ構成は以下の通り．

```bash: tree
.
├── README.md
├── develop
│   ├── data
│   ├── logs
│   ├── models
│   ├── outputs
│   └── src
├── docker-compose.yml
└── environment
    ├── .config
    │   └── ***
    └── Dockerfile.mlenv
```

- `develop`: 開発関連のファイルを格納するディレクトリ
    - `./data`: データセットを格納するディレクトリ
    - `./logs`: ログファイルを格納するディレクトリ
    - `./models`: モデルを格納するディレクトリ
    - `./outputs`: 出力ファイルを格納するディレクトリ
    - `./src`: ソースコードを格納するディレクトリ
- `environment`: 環境関連のファイルを格納するディレクトリ
    - `.config`: `JupyterLab`の設定ファイルを格納するディレクトリ


# Quick Start
1. テンプレートの使用
    - `Create a new repository` > `Repository template`から本リポジトリを選択
3. リポジトリのクローン
    ```bash
    git clone <repository_url>
    ```
4. Dockerコンテナの起動
    ```bash
    docker compose up --build -d
    ```
