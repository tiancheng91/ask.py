# ask.py

[![PyPI version](https://badge.fury.io/py/ask-py-cli.svg)](https://badge.fury.io/py/ask-py-cli)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

English | [简体中文](README.md) | [日本語](README_JA.md)

A terminal LLM Q&A tool based on LangChain, with multi-model support, role memory, and MCP tool integration.

## Features

- 🚀 Quick terminal Q&A - just `ask "question"`
- 🔧 Multi-model configuration - supports any OpenAI-compatible API
- 🎭 Role system with custom System Prompts
- 🧠 Three-tier memory system (short/medium/long-term) with auto compression
- 🔌 MCP (Model Context Protocol) tool support

## Quick Start

### 1. Install

```bash
# Using pipx (recommended)
pipx install ask-py-cli

# Or using uv tool
uv tool install ask-py-cli
```

### 2. Add a Model

```bash
ask model add openai \
    -b https://api.openai.com/v1 \
    -k $OPENAI_API_KEY \
    -m gpt-4 \
    --set-default
```

### 3. Start Using

```bash
# Ask a question
ask "What is quantum computing?"

# Use tool mode (time queries, shell commands, etc.)
ask -t "What time is it?"
ask -t "List files in /tmp"
ask -t "Organize video filenames in ~/Downloads"

# Create a role (with memory)
ask role add coder -s "You are a senior programmer" --set-default
ask "Write a quicksort"
ask "Convert it to iterative version"  # Automatically remembers context
```

## Command Reference

### Asking Questions

```bash
ask [OPTIONS] "question"

Options:
  -m, --model TEXT   Specify model name
  -s, --system TEXT  Set temporary system prompt
  -r, --role TEXT    Use specified role
  -t, --tools        Enable MCP tools
  --mcp NAME         Specify MCP server (can be used multiple times)
```

### Model Management

```bash
ask model add NAME -b API_BASE -k API_KEY [-m MODEL] [--set-default]
ask model list
ask model default NAME
ask model remove NAME
```

### Role Management

```bash
ask role add NAME -s "prompt" [-m MODEL] [--set-default]
ask role list
ask role show NAME
ask role edit NAME -s "new prompt"
ask role default [NAME]      # Set/clear default role
ask role remove NAME
ask role memory NAME         # View memory
ask role clear-memory NAME --confirm
```

## Configuration Files

Configurations are stored in `~/.config/ask/`:

```
~/.config/ask/
├── config.yaml    # Model configuration
├── roles.yaml     # Role configuration
├── mcp.json       # MCP server configuration
└── memory/        # Memory storage
```

### config.yaml Example

```yaml
default: openai
default_role: coder
lang: en  # Language: en, zh-cn, zh-tw, ja (auto-detects from $LANG by default)
models:
  openai:
    api_base: https://api.openai.com/v1
    api_key: sk-xxx
    model: gpt-4
    temperature: 0.7
```

### Multi-language Support

Supported languages:
- `en` - English
- `zh-cn` - Simplified Chinese
- `zh-tw` - Traditional Chinese
- `ja` - Japanese

Language detection priority:
1. `lang` setting in config file
2. Environment variable `$LANG`
3. Defaults to English

## Memory System

Roles support three-tier hierarchical memory with automatic conversation history management:

| Tier | Description | Strategy |
|------|-------------|----------|
| Short-term | Recent complete conversations | Keep 10 rounds |
| Medium-term | Earlier conversation summaries | LLM compressed |
| Long-term | Overall refined summary | Multiple summaries merged |

## MCP Tool Support

MCP (Model Context Protocol) enables LLM to call external tools.

> ⚠️ **Note**: Tool mode requires spawning external processes, which is slower. Use `-t` only when needed.

### Default Configuration

Auto-created on first run at `~/.config/ask/mcp.json`, automatically detects `uvx` or `pipx`:

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

- `time`: Query current time
- `shell`: Execute system commands (restricted by `ALLOW_COMMANDS`)
- Auto-detection: prefers `uvx`, falls back to `pipx`

### Adding More Servers

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

> MCP tools are invoked dynamically via `uvx` or `npx`. Requires [uv](https://docs.astral.sh/uv/) or Node.js.

### Using Tools

```bash
ask mcp list              # List servers
ask mcp tools shell       # View shell tool details
ask -t "What time is it?" # Use default enabled tools (time + shell)
ask -t "List files in /tmp"              # LLM auto-invokes shell
ask -t "Organize video filenames in ~/Videos" # LLM plans and executes commands
```

### Role-level MCP

```yaml
# ~/.config/ask/roles.yaml
coder:
  system_prompt: "You are a programmer"
  mcp: ["github"]  # Additional servers to enable
```

## Supported Models

Any OpenAI-compatible API: OpenAI, Azure OpenAI, DeepSeek, GLM, Ollama, vLLM, LM Studio, etc.

## Development

```bash
# Clone and install
git clone https://github.com/tiancheng91/ask.py
cd ask.py
uv sync

# Run
uv run ask "question"

# Test
uv run pytest test_ask.py -v

# Build and publish
uv build
uv publish
```

### Install from Source

```bash
pipx install git+https://github.com/tiancheng91/ask.py
# Or
uv tool install git+https://github.com/tiancheng91/ask.py
# Or from PyPI
pipx install ask-py-cli
```

## License

MIT
