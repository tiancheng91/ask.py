# ask.py

[![PyPI version](https://badge.fury.io/py/ask-py-cli.svg)](https://badge.fury.io/py/ask-py-cli)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[English](README_EN.md) | [简体中文](README.md) | 日本語

LangChain ベースのターミナル LLM Q&A ツール。マルチモデル、ロールメモリ、MCP ツール連携をサポート。

## 特徴

- 🚀 クイックターミナル Q&A - `ask "質問"` だけで OK
- 🔧 マルチモデル設定 - OpenAI 互換 API をサポート
- 🎭 カスタム System Prompt 付きロールシステム
- 🧠 三層メモリシステム（短期/中期/長期）、自動圧縮機能付き
- 🔌 MCP（Model Context Protocol）ツールサポート

## クイックスタート

### 1. インストール

```bash
# pipx を使用（推奨）
pipx install ask-py-cli

# または uv tool を使用
uv tool install ask-py-cli
```

### 2. モデルを追加

```bash
ask model add openai \
    -b https://api.openai.com/v1 \
    -k $OPENAI_API_KEY \
    -m gpt-4 \
    --set-default
```

### 3. 使い始める

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

```yaml
default: openai
default_role: coder
lang: ja  # 言語: en, zh-cn, zh-tw, ja（デフォルトは $LANG から自動検出）
models:
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

## メモリシステム

ロールは三層階層メモリをサポートし、会話履歴を自動管理します：

| 層 | 説明 | 戦略 |
|----|------|------|
| 短期 | 最近の完全な会話 | 10 ラウンド保持 |
| 中期 | 以前の会話要約 | LLM で圧縮 |
| 長期 | 全体の洗練された要約 | 複数の要約をマージ |

## MCP ツールサポート

MCP（Model Context Protocol）により、LLM が外部ツールを呼び出せます。

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
        "ALLOW_COMMANDS": "ls,cat,head,tail,find,grep,wc,pwd,echo,mkdir,cp,mv,touch,date"
      }
    }
  },
  "enabled": ["time", "shell"]
}
```

- `time`: 現在時刻を取得
- `shell`: システムコマンドを実行（`ALLOW_COMMANDS` で制限）
- 自動検出: `uvx` を優先、なければ `pipx` を使用

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
ask -t "今何時？"          # デフォルト有効ツールを使用（time + shell）
ask -t "/tmp のファイル一覧"           # LLM が自動で shell を呼び出す
ask -t "~/Videos の動画ファイル名を整理" # LLM が計画してコマンドを実行
```

### ロールレベル MCP

```yaml
# ~/.config/ask/roles.yaml
coder:
  system_prompt: "あなたはプログラマーです"
  mcp: ["github"]  # 追加で有効にするサーバー
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
