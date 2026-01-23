# ask.py

[![PyPI version](https://badge.fury.io/py/ask-py-cli.svg)](https://badge.fury.io/py/ask-py-cli)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

English | [简体中文](README.md) | [日本語](README_JA.md)

A terminal LLM Q&A tool based on LangChain, with multi-model support, role memory, and MCP tool integration.

## Why Choose ask.py

Compared to traditional AI Agent frameworks, `ask.py` focuses on a **lightweight, fast, and terminal-friendly** design:

- **Extremely Lightweight**: No complex Agent framework required, low resource usage, fast response
- **Terminal-First**: Get answers with `ask "question"`, streaming output, seamlessly integrated into terminal workflows
- **Focused on Core Scenarios**: Perfect for quick queries, code analysis, and error diagnosis in daily development

Enjoy a lightweight AI Q&A experience without starting heavy Agent frameworks.

## Features

- 🚀 Quick terminal Q&A - just `ask "question"`
- ⚡ Streaming output - real-time response display (enabled by default)
- 🔧 Multi-model configuration - supports any OpenAI-compatible API
- 🎭 Role system with custom System Prompts
- 🧠 Three-tier memory system (short/medium/long-term) with auto compression
- 🔌 MCP (Model Context Protocol) tool support
- 📁 File content analysis - supports `-f` parameter to read files
- 📊 Error log analysis - supports reading from stdin
- 🌍 Context awareness - auto-injects working directory, environment variables, etc.

## Quick Start

### 1. Install

```bash
# Using pipx (recommended)
pipx install ask-py-cli

# Or using uv tool
uv tool install ask-py-cli
```

### 2. Start Using

After installation, you can use it directly. Default configuration is auto-created on first run:

```bash
# Ask a question (using default public model)
ask "What is Python?"
```

> ⚠️ **Important Notes**: 
> - The default `public(glm-4-flash)` model is for **quick experience only**, with **IP-based rate limiting** (dynamically adjusted)
> - **For long-term use, configure your own API key** to avoid rate limit restrictions
> - Supports any OpenAI-compatible API: OpenAI, Azure OpenAI, DeepSeek, GLM, Ollama, etc.

#### Add Your Own Model

```bash
ask model add openai \
    -b https://api.openai.com/v1 \
    -k $OPENAI_API_KEY \
    -m gpt-4 \
    --set-default
```

### 3. Usage Examples

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
  -m, --model TEXT    Specify model name
  -s, --system TEXT   Set temporary system prompt
  -r, --role TEXT     Use specified role
  -t, --tools         Enable MCP tools
  --mcp NAME          Specify MCP server (can be used multiple times)
  --no-stream         Disable streaming output (show complete result at once)
  -f, --file TEXT     Read file content and analyze
  --stdin             Read additional input from stdin (for error analysis, etc.)
```

### Usage Examples

#### Daily Q&A
```bash
# Quick questions (streaming output, real-time display)
ask "What is Python's generator?"
ask "Explain RESTful API design principles"
```

#### Code Analysis
```bash
# Analyze single file
ask -f main.py "Explain this file's functionality"
ask "Optimize this code's performance" -f utils.py

# Analyze configuration files
ask -f docker-compose.yml "Check if configuration is correct"
ask -f package.json "Explain dependencies"
```

#### Error Diagnosis
```bash
# Analyze error logs
cat error.log | ask "Analyze this error" --stdin
python script.py 2>&1 | ask "Explain this error" --stdin

# Analyze application logs
tail -n 100 app.log | ask "Find performance bottlenecks" --stdin
journalctl -u myapp -n 50 | ask "Analyze service issues" --stdin
```

#### System Administration
```bash
# Use shell role for system management
ask role add shell -s "You are a system administrator assistant. When users ask about system-related questions (such as file operations, process management, system information queries, etc.), prioritize using shell commands to solve them rather than implementing with other programming languages." --set-default
ask "Find the directory with largest disk usage"
ask "List all listening ports"  # Automatically remembers context

# System problem diagnosis
ask -r shell "Clean files older than 7 days in /tmp"
ask -r shell "Check system load and find the cause"
```

#### Tool Integration
```bash
# Use MCP tools
ask -t "What time is it?"  # Query time
ask --mcp shell "List Python files in current directory"

# Combine usage
ask -f requirements.txt "Check dependency conflicts" --no-stream | tee analysis.txt
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

Default configuration is auto-created on first run:

```yaml
default: public
lang: en  # Language: en, zh-cn, zh-tw, ja (auto-detects from $LANG by default)
models:
  public:
    api_base: https://ask.appsvc.net/v1
    api_key: <auto-generated key>
    model: glm-4-flash
    temperature: 0.7
```

> ⚠️ **Important Notes**: 
> - The `public(glm-4-flash)` model is for quick experience only, with IP-based rate limiting (dynamically adjusted)
> - For long-term use, add your own model configuration with your own API key
> - You can add multiple models and switch default model with `ask model default <name>`

After adding your own models:

```yaml
default: openai
default_role: shell
lang: en
models:
  public:
    api_base: https://ask.appsvc.net/v1
    api_key: <auto-generated key>
    model: glm-4-flash
    temperature: 0.7
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

## Core Features

### Streaming Output

Streaming output is enabled by default, displaying responses in real-time:

```bash
ask "Explain quantum computing principles"  # Real-time display, no need to wait for complete response
ask "Detailed explanation" --no-stream  # Disable streaming, show complete result at once
```

### Context Awareness

Automatically injects current environment information for more contextual responses:

- Current working directory
- Operating system and Python version
- Important environment variables (PATH, HOME, USER, SHELL, LANG, etc.)

No manual configuration needed, the system automatically identifies and adds to context.

### File Content Analysis

Directly analyze code files, configuration files, etc.:

```bash
ask -f main.py "Explain this file"
ask "Optimize this code" -f utils.py
ask -f config.yaml "Check if configuration is correct"
```

### Error Log Analysis

Read error information from standard input for analysis:

```bash
# Analyze error logs
cat error.log | ask "Analyze this error" --stdin

# Analyze recent logs
tail -n 100 app.log | ask "Find the cause of the problem" --stdin

# Analyze command output
python script.py 2>&1 | ask "Explain this error" --stdin
```

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
        "ALLOW_COMMANDS": "ls,cat,head,tail,find,grep,wc,pwd,echo,mkdir,cp,mv,touch,date,whoami,hostname,ps,du"
      }
    }
  },
  "enabled": ["time"]
}
```

- `time`: Query current time (enabled by default)
- `shell`: Execute system commands (restricted by `ALLOW_COMMANDS`, not enabled by default)
- Auto-detection: prefers `uvx`, falls back to `pipx`

> ⚠️ **Note**: The `shell` server is not enabled by default due to execution accuracy issues. To use it, specify manually with `--mcp shell`, or edit the config file to add `"shell"` to `enabled`.

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
ask -t "What time is it?" # Use default enabled tools (time)
ask -t "List files in /tmp"              # Need to manually enable shell server
ask --mcp shell "List files in current directory"  # Manually specify shell server
```

> ⚠️ **Note**: The `shell` server is not enabled by default due to execution accuracy issues. To use it, specify manually with `--mcp shell`, or edit the config file to enable it.

### Role-level MCP

```yaml
# ~/.config/ask/roles.yaml
shell:
  system_prompt: "You are a system administrator assistant. When users ask about system-related questions (such as file operations, process management, system information queries, etc.), prioritize using shell commands to solve them rather than implementing with other programming languages."
  mcp: ["shell"]  # Enable shell server to execute commands
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
