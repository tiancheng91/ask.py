# ask.py

[![PyPI version](https://badge.fury.io/py/ask-py-cli.svg)](https://badge.fury.io/py/ask-py-cli)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

基于 LangChain 的终端 LLM 问答工具，支持多模型、角色记忆和 MCP 工具调用。

## 特性

- 🚀 快速终端问答，直接 `ask "问题"` 即可
- 🔧 多模型配置，支持任意 OpenAI 兼容接口
- 🎭 角色系统，自定义 System Prompt
- 🧠 三层记忆系统（短期/中期/长期），自动压缩淘汰
- 🔌 MCP（Model Context Protocol）工具支持

## 快速开始

### 1. 安装

```bash
# 使用 pipx（推荐）
pipx install ask-py-cli

# 或使用 uv tool
uv tool install ask-py-cli
```

### 2. 配置模型

```bash
ask config add openai \
    -b https://api.openai.com/v1 \
    -k $OPENAI_API_KEY \
    -m gpt-4 \
    --set-default
```

### 3. 开始使用

```bash
# 直接提问
ask "什么是量子计算？"

# 使用 MCP 工具
ask -t "现在几点了？"

# 创建角色（带记忆）
ask role add coder -s "你是一个资深程序员" --set-default
ask "写一个快速排序"
ask "改成迭代版本"  # 自动记忆上下文
```

## 命令参考

### 提问

```bash
ask [OPTIONS] "问题"

选项:
  -m, --model TEXT   指定模型
  -s, --system TEXT  临时系统提示词
  -r, --role TEXT    使用指定角色
  -t, --tools        启用 MCP 工具
  --mcp NAME         指定 MCP 服务器（可多次使用）
```

### 模型管理

```bash
ask config add NAME -b API_BASE -k API_KEY [-m MODEL] [--set-default]
ask config list
ask config default NAME
ask config remove NAME
```

### 角色管理

```bash
ask role add NAME -s "提示词" [-m MODEL] [--set-default]
ask role list
ask role show NAME
ask role edit NAME -s "新提示词"
ask role default [NAME]      # 设置/清除默认角色
ask role remove NAME
ask role memory NAME         # 查看记忆
ask role clear-memory NAME --confirm
```

## 配置文件

配置存储在 `~/.config/ask/` 目录：

```
~/.config/ask/
├── config.yaml    # 模型配置
├── roles.yaml     # 角色配置
├── mcp.json       # MCP 服务器配置
└── memory/        # 记忆存储
```

### config.yaml 示例

```yaml
default: openai
default_role: coder
models:
  openai:
    api_base: https://api.openai.com/v1
    api_key: sk-xxx
    model: gpt-4
    temperature: 0.7
```

## 记忆系统

角色支持三层分层记忆，自动管理对话历史：

| 层级 | 说明 | 策略 |
|------|------|------|
| 短期 | 最近完整对话 | 保留 10 轮 |
| 中期 | 早期对话摘要 | LLM 压缩生成 |
| 长期 | 整体精炼总结 | 多摘要合并 |

## MCP 工具支持

MCP（Model Context Protocol）让 LLM 能够调用外部工具。

### 默认配置

首次运行自动创建 `~/.config/ask/mcp.json`：

```json
{
  "mcpServers": {
    "time": {
      "command": "uvx",
      "args": ["mcp-server-time"]
    }
  },
  "enabled": ["time"]
}
```

### 添加更多服务器

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

> MCP 工具通过 `uvx` 或 `npx` 动态调用，需安装 [uv](https://docs.astral.sh/uv/) 或 Node.js。

### 使用工具

```bash
ask mcp list              # 查看服务器
ask mcp tools time        # 查看工具列表
ask -t "现在几点？"        # 使用默认启用的工具
ask --mcp fetch "获取网页" # 指定服务器
```

### 角色级 MCP

```yaml
# ~/.config/ask/roles.yaml
coder:
  system_prompt: "你是一个程序员"
  mcp: ["github"]  # 额外启用的服务器
```

## 支持的模型

任何 OpenAI 兼容接口：OpenAI、Azure OpenAI、DeepSeek、智谱 GLM、Ollama、vLLM、LM Studio 等。

## 开发

```bash
# 克隆并安装
git clone https://github.com/tiancheng91/ask.py
cd ask.py
uv sync

# 运行
uv run ask "问题"

# 测试
uv run pytest test_ask.py -v

# 构建发布
uv build
uv publish
```

### 从源码安装

```bash
pipx install git+https://github.com/tiancheng91/ask.py
# 或
uv tool install git+https://github.com/tiancheng91/ask.py
# 或从 PyPI 安装
pipx install ask-py-cli
```

## License

MIT
