# ask.py

[![PyPI version](https://badge.fury.io/py/ask-py-cli.svg)](https://badge.fury.io/py/ask-py-cli)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[English](README_EN.md) | [简体中文](README.md) | 日本語

ターミナル LLM Q&A ツール。マルチモデル、ロールメモリ、MCP ツール連携をサポート。

## ask.py を選ぶ理由

従来の AI Agent フレームワークと比較して、`ask.py` は**軽量、高速、ターミナル重視**の設計に焦点を当てています：

- **極めて軽量**：複雑な Agent フレームワーク不要、リソース使用量が少なく、高速レスポンス
- **ターミナル優先**：`ask "質問"` で回答を取得、ストリーミング出力、ターミナルワークフローにシームレスに統合
- **コアシナリオに特化**：日常開発におけるクイッククエリ、コード分析、エラー診断に最適

重い Agent フレームワークを起動することなく、軽量な AI Q&A 体験を楽しめます。

## 特徴

- 🚀 クイックターミナル Q&A - `ask "質問"` だけで OK
- ⚡ ストリーミング出力 - リアルタイムで回答を表示（デフォルトで有効）
- 🔧 マルチモデル設定 - OpenAI 互換 API をサポート
- 🎭 カスタム System Prompt 付きロールシステム
- 🧠 三層メモリシステム（短期/中期/長期）、自動圧縮機能付き
- 🔌 MCP（Model Context Protocol）ツールサポート
- 📁 ファイル内容分析 - `-f` パラメータでファイルを読み取り
- 📊 エラーログ分析 - stdin から読み取りをサポート
- 🌍 コンテキスト認識 - 作業ディレクトリ、環境変数などを自動注入

## クイックスタート

### 1. インストール

```bash
# pipx を使用（推奨）
pipx install ask-py-cli

# または uv tool を使用
uv tool install ask-py-cli
```

### 2. 使い始める

インストール後すぐに使用できます。初回実行時にデフォルト設定が自動作成されます：

```bash
# 質問する（デフォルトの public モデルを使用）
ask "Python とは？"
```

> ⚠️ **重要な注意事項**: 
> - デフォルトの `public(glm-4-flash)` モデルは**クイック体験専用**で、**IP ベースのレート制限**があります（動的に調整）
> - **長期使用には独自の API キーを設定することを推奨**します。レート制限の影響を避けるため
> - あらゆる OpenAI 互換 API をサポート：OpenAI、Azure OpenAI、DeepSeek、GLM、Ollama など

#### 独自のモデルを追加

```bash
ask model add openai \
    -b https://api.openai.com/v1 \
    -k $OPENAI_API_KEY \
    -m gpt-4 \
    --set-default
```

### 3. 使用例

```bash
# 質問する
ask "量子コンピューティングとは？"

# ツールモード（時間クエリ、シェルコマンドなど）
ask -t "今何時？"
ask -t "/tmp のファイルを一覧表示"
ask -t "~/Downloads の動画ファイル名を整理"

# ロールを作成（メモリ付き）
ask role add coder -s "あなたはシニアプログラマーです" --set-default
ask "クイックソートを書いて"
ask "イテレーティブ版に変換して"  # コンテキストを自動記憶
```

## コマンドリファレンス

### 質問する

```bash
ask [OPTIONS] "質問"

オプション:
  -m, --model TEXT   モデル名を指定
  -s, --system TEXT  一時的なシステムプロンプトを設定
  -r, --role TEXT    指定したロールを使用
  -t, --tools        MCP ツールを有効化
  --mcp NAME         MCP サーバーを指定（複数回使用可）
  --no-stream        ストリーミング出力を無効化（完全な結果を一度に表示）
  -f, --file TEXT    ファイル内容を読み取って分析
  --stdin            stdin から追加の入力を読み取る（エラー分析など）
```

### 使用例

#### 日常的な Q&A
```bash
# クイック質問（ストリーミング出力、リアルタイム表示）
ask "Python のジェネレータとは？"
ask "RESTful API 設計原則を説明"
```

#### コード分析
```bash
# 単一ファイルを分析
ask -f main.py "このファイルの機能を説明"
ask "このコードのパフォーマンスを最適化" -f utils.py

# 設定ファイルを分析
ask -f docker-compose.yml "設定が正しいか確認"
ask -f package.json "依存関係を説明"
```

#### エラー診断
```bash
# エラーログを分析
cat error.log | ask "このエラーを分析" --stdin
python script.py 2>&1 | ask "このエラーを説明" --stdin

# アプリケーションログを分析
tail -n 100 app.log | ask "パフォーマンスのボトルネックを特定" --stdin
journalctl -u myapp -n 50 | ask "サービス問題を分析" --stdin
```

#### システム管理支援
```bash
# シェルロールを使用してシステム管理対話
ask role add shell -s "あなたはシステム管理者アシスタントです。ユーザーがシステム関連の質問（ファイル操作、プロセス管理、システム情報クエリなど）をした場合、他のプログラミング言語コードで実装するのではなく、シェルコマンドを使用して解決することを優先します。" --set-default
ask "ディスク使用量が最も大きいディレクトリを特定"
ask "すべてのリスニングポートをリスト"  # コンテキストを自動記憶

# システム問題の診断
ask -r shell "/tmp ディレクトリで 7 日を超えるファイルをクリーンアップ"
ask -r shell "システム負荷を確認して原因を特定"
```

#### ツール統合
```bash
# MCP ツールを使用
ask -t "今何時？"  # 時刻をクエリ
ask --mcp shell "現在のディレクトリの Python ファイルをリスト"

# 組み合わせて使用
ask -f requirements.txt "依存関係の競合を確認" --no-stream | tee analysis.txt
```

### モデル管理

```bash
ask model add NAME -b API_BASE -k API_KEY [-m MODEL] [--set-default]
ask model list
ask model default NAME
ask model remove NAME
```

### ロール管理

```bash
ask role add NAME -s "プロンプト" [-m MODEL] [--set-default]
ask role list
ask role show NAME
ask role edit NAME -s "新しいプロンプト"
ask role default [NAME]      # デフォルトロールを設定/クリア
ask role remove NAME
ask role memory NAME         # メモリを表示
ask role clear-memory NAME --confirm
```

## 設定ファイル

設定は `~/.config/ask/` に保存されます：

```
~/.config/ask/
├── config.yaml    # モデル設定
├── roles.yaml     # ロール設定
├── mcp.json       # MCP サーバー設定
└── memory/        # メモリストレージ
```

### config.yaml の例

初回実行時にデフォルト設定が自動作成されます：

```yaml
default: public
lang: ja  # 言語: en, zh-cn, zh-tw, ja（デフォルトは $LANG から自動検出）
models:
  public:
    api_base: https://ask.appsvc.net/v1
    api_key: <自動生成されたキー>
    model: glm-4-flash
    temperature: 0.7
```

> ⚠️ **重要な注意事項**: 
> - `public(glm-4-flash)` モデルはクイック体験専用で、IP ベースのレート制限があります（動的に調整）
> - 長期使用には独自のモデル設定を追加し、独自の API キーを使用することを推奨します
> - 複数のモデルを追加でき、`ask model default <name>` でデフォルトモデルを切り替えられます

独自のモデルを追加した後：

```yaml
default: openai
default_role: shell
lang: ja
models:
  public:
    api_base: https://ask.appsvc.net/v1
    api_key: <自動生成されたキー>
    model: glm-4-flash
    temperature: 0.7
  openai:
    api_base: https://api.openai.com/v1
    api_key: sk-xxx
    model: gpt-4
    temperature: 0.7
```

### 多言語サポート

サポート言語：
- `en` - English
- `zh-cn` - 简体中文
- `zh-tw` - 繁體中文
- `ja` - 日本語

言語検出の優先順位：
1. 設定ファイルの `lang` 設定
2. 環境変数 `$LANG`
3. デフォルトは英語

## コア機能

### ストリーミング出力

デフォルトでストリーミング出力が有効で、回答内容をリアルタイムで表示します：

```bash
ask "量子コンピューティングの原理を説明"  # リアルタイム表示、完全な応答を待つ必要なし
ask "詳細な説明" --no-stream  # ストリーミングを無効化、完全な結果を一度に表示
```

### コンテキスト認識

現在の環境情報を自動注入し、より実際のシナリオに合った回答を提供します：

- 現在の作業ディレクトリ
- オペレーティングシステムと Python バージョン
- 重要な環境変数（PATH, HOME, USER, SHELL, LANG など）

手動設定は不要で、システムが自動的に識別してコンテキストに追加します。

### ファイル内容分析

コードファイル、設定ファイルなどを直接分析します：

```bash
ask -f main.py "このファイルを説明"
ask "このコードを最適化" -f utils.py
ask -f config.yaml "設定が正しいか確認"
```

### エラーログ分析

標準入力からエラー情報を読み取って分析します：

```bash
# エラーログを分析
cat error.log | ask "このエラーを分析" --stdin

# 最近のログを分析
tail -n 100 app.log | ask "問題の原因を特定" --stdin

# コマンド出力を分析
python script.py 2>&1 | ask "このエラーを説明" --stdin
```

## メモリシステム

ロールは三層階層メモリをサポートし、会話履歴を自動管理します：

| 層 | 説明 | 戦略 |
|----|------|------|
| 短期 | 最近の完全な会話 | 10 ラウンド保持 |
| 中期 | 以前の会話要約 | LLM で圧縮 |
| 長期 | 全体の洗練された要約 | 複数の要約をマージ |

## MCP ツールサポート

MCP（Model Context Protocol）により、LLM が外部ツールを呼び出せます。

> ⚠️ **注意**: ツールモードは外部プロセスを起動するため、応答が遅くなります。必要な時のみ `-t` を使用してください。

### デフォルト設定

初回実行時に `~/.config/ask/mcp.json` が自動作成され、`uvx` または `pipx` を自動検出します：

```json
{
  "mcpServers": {
    "time": {
      "command": "uvx",
      "args": ["mcp-server-time"]
    },
    "shell": {
      "command": "uvx",
      "args": ["mcp-shell-server"],
      "env": {
        "ALLOW_COMMANDS": "ls,cat,head,tail,find,grep,wc,pwd,echo,mkdir,cp,mv,touch,date,whoami,hostname,ps,du"
      }
    }
  },
  "enabled": ["time"]
}
```

- `time`: 現在時刻を取得（デフォルトで有効）
- `shell`: システムコマンドを実行（`ALLOW_COMMANDS` で制限、デフォルトで無効）
- 自動検出: `uvx` を優先、なければ `pipx` を使用

> ⚠️ **注意**: `shell` サーバーは実行精度の問題により、デフォルトで無効になっています。使用するには、`--mcp shell` で手動指定するか、設定ファイルの `enabled` に `"shell"` を追加してください。

### サーバーを追加

```json
{
  "mcpServers": {
    "time": { "command": "uvx", "args": ["mcp-server-time"] },
    "filesystem": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"] },
    "fetch": { "command": "uvx", "args": ["mcp-server-fetch"] }
  },
  "enabled": ["time", "filesystem"]
}
```

> MCP ツールは `uvx` または `npx` で動的に呼び出されます。[uv](https://docs.astral.sh/uv/) または Node.js が必要です。

### ツールを使用

```bash
ask mcp list              # サーバー一覧
ask mcp tools shell       # shell ツールの詳細を表示
ask -t "今何時？"          # デフォルト有効ツールを使用（time）
ask -t "/tmp のファイル一覧"           # shell サーバーを手動で有効化する必要がある
ask --mcp shell "現在のディレクトリのファイルをリスト"  # shell サーバーを手動指定
```

> ⚠️ **注意**: `shell` サーバーは実行精度の問題により、デフォルトで無効になっています。使用するには、`--mcp shell` で手動指定するか、設定ファイルで有効化してください。

### ロールレベル MCP

```yaml
# ~/.config/ask/roles.yaml
shell:
  system_prompt: "あなたはシステム管理者アシスタントです。ユーザーがシステム関連の質問（ファイル操作、プロセス管理、システム情報クエリなど）をした場合、他のプログラミング言語コードで実装するのではなく、シェルコマンドを使用して解決することを優先します。"
  mcp: ["shell"]  # コマンドを実行するために shell サーバーを有効化
```

## サポートモデル

OpenAI 互換 API なら何でも：OpenAI、Azure OpenAI、DeepSeek、GLM、Ollama、vLLM、LM Studio など。

## 開発

```bash
# クローンしてインストール
git clone https://github.com/tiancheng91/ask.py
cd ask.py
uv sync

# 実行
uv run ask "質問"

# テスト
uv run pytest test_ask.py -v

# ビルドと公開
uv build
uv publish
```

### ソースからインストール

```bash
pipx install git+https://github.com/tiancheng91/ask.py
# または
uv tool install git+https://github.com/tiancheng91/ask.py
# PyPI からインストール
pipx install ask-py-cli
```

## ライセンス

MIT
