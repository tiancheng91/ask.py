#!/usr/bin/env python3
"""
ask.py - 基于 LangChain 的终端 LLM 问答工具
支持配置多个 OpenAI 兼容的模型接入点
支持 Role（角色）和分层记忆功能
支持 MCP（Model Context Protocol）工具调用
"""

# 在任何其他导入之前关闭 urllib3 SSL 警告
import warnings
warnings.filterwarnings("ignore", message=".*urllib3.*OpenSSL.*")
warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import click
import yaml
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# 配置文件路径
CONFIG_DIR = Path.home() / ".config" / "ask"
CONFIG_FILE = CONFIG_DIR / "config.yaml"
ROLES_FILE = CONFIG_DIR / "roles.yaml"
MEMORY_DIR = CONFIG_DIR / "memory"
MCP_FILE = CONFIG_DIR / "mcp.json"

console = Console()

# MCP 可用性标志
MCP_AVAILABLE = False
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    pass

# ==================== MCP 管理 ====================

DEFAULT_MCP_CONFIG = {
    "mcpServers": {
        "time": {
            "command": "uvx",
            "args": ["mcp-server-time"]
        }
    },
    "enabled": ["time"]
}


def load_mcp_config() -> dict:
    """加载 MCP 服务器配置 (JSON 格式)
    
    配置文件: ~/.config/ask/mcp.json
    
    格式与 Claude Desktop 兼容:
    ```json
    {
      "mcpServers": {
        "time": {
          "command": "uvx",
          "args": ["mcp-server-time"]
        },
        "filesystem": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        },
        "fetch": {
          "command": "uvx",
          "args": ["mcp-server-fetch"]
        }
      },
      "enabled": ["time", "filesystem"]
    }
    ```
    
    - mcpServers: MCP 服务器定义（与 Claude Desktop 格式兼容）
    - enabled: 全局启用的服务器列表（可选，不设置则全部启用）
    """
    if not MCP_FILE.exists():
        # 首次使用时创建默认配置
        save_mcp_config(DEFAULT_MCP_CONFIG)
        return DEFAULT_MCP_CONFIG.copy()
    
    with open(MCP_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    return {
        "mcpServers": config.get("mcpServers", {}),
        "enabled": config.get("enabled"),  # None 表示全部启用
    }


def save_mcp_config(config: dict) -> None:
    """保存 MCP 配置"""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(MCP_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def get_available_mcp_servers(role_name: Optional[str] = None) -> List[str]:
    """获取可用的 MCP 服务器列表
    
    优先级: 命令行指定 > 角色配置 > 全局配置 > 全部服务器
    
    Args:
        role_name: 角色名称
    
    Returns:
        可用的 MCP 服务器名称列表
    """
    mcp_config = load_mcp_config()
    all_servers = list(mcp_config.get("mcpServers", {}).keys())
    
    if not all_servers:
        return []
    
    # 全局启用的服务器
    global_enabled = mcp_config.get("enabled")
    if global_enabled is None:
        global_enabled = all_servers  # 未配置则全部启用
    
    # 如果有角色，合并角色配置
    if role_name:
        roles = load_roles()
        role = roles.get(role_name, {})
        role_mcp = role.get("mcp", [])
        
        if role_mcp:
            # 合并: 全局 + 角色特有（去重，保持顺序）
            merged = list(global_enabled)
            for s in role_mcp:
                if s not in merged and s in all_servers:
                    merged.append(s)
            return merged
    
    return [s for s in global_enabled if s in all_servers]


def get_mcp_server_by_name(name: str) -> Optional[dict]:
    """根据名称获取 MCP 服务器配置"""
    mcp_config = load_mcp_config()
    return mcp_config.get("mcpServers", {}).get(name)


async def get_mcp_tools(server_name: str, server_config: dict) -> tuple:
    """连接 MCP 服务器并获取工具列表"""
    if server_config.get("type") == "sse":
        raise RuntimeError("SSE 类型暂不支持，请使用 stdio 类型")
    
    command = server_config.get("command")
    args = server_config.get("args", [])
    env = server_config.get("env")
    
    server_params = StdioServerParameters(
        command=command,
        args=args,
        env=env
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            return session, tools_result.tools


async def call_mcp_tool(server_config: dict, tool_name: str, arguments: dict) -> Any:
    """调用 MCP 工具"""
    command = server_config.get("command")
    args = server_config.get("args", [])
    env = server_config.get("env")
    
    server_params = StdioServerParameters(
        command=command,
        args=args,
        env=env
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            return result


def convert_mcp_tools_to_openai(tools: list) -> list:
    """将 MCP 工具转换为 OpenAI function calling 格式"""
    openai_tools = []
    for tool in tools:
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema if tool.inputSchema else {"type": "object", "properties": {}}
            }
        }
        openai_tools.append(openai_tool)
    return openai_tools


async def run_with_mcp_tools(
    question: str,
    llm: ChatOpenAI,
    server_names: Optional[List[str]] = None,
    system_prompt: Optional[str] = None,
    role_name: Optional[str] = None
) -> str:
    """使用 MCP 工具执行问答（ReAct 模式）"""
    if not MCP_AVAILABLE:
        raise RuntimeError("MCP 支持未安装，请运行: uv sync")
    
    # 确定要使用的服务器
    if server_names:
        servers_to_use = server_names
    else:
        servers_to_use = get_available_mcp_servers(role_name)
    
    if not servers_to_use:
        raise RuntimeError("没有可用的 MCP 服务器，请配置 ~/.config/ask/mcp.json")
    
    # 目前只支持单个服务器
    server_name = servers_to_use[0]
    server_config = get_mcp_server_by_name(server_name)
    
    if not server_config:
        raise RuntimeError(f"MCP 服务器 '{server_name}' 不存在")
    
    command = server_config.get("command")
    args = server_config.get("args", [])
    env = server_config.get("env")
    
    server_params = StdioServerParameters(
        command=command,
        args=args,
        env=env
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # 获取工具列表
            tools_result = await session.list_tools()
            mcp_tools = tools_result.tools
            
            if not mcp_tools:
                raise RuntimeError("MCP 服务器没有提供任何工具")
            
            # 转换为 OpenAI 格式
            openai_tools = convert_mcp_tools_to_openai(mcp_tools)
            
            # 构建消息
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": question})
            
            # 循环执行工具调用（ReAct 模式）
            max_iterations = 10
            for _ in range(max_iterations):
                # 调用 LLM
                response = llm.invoke(
                    messages,
                    tools=openai_tools,
                    tool_choice="auto"
                )
                
                # 检查是否有工具调用
                if not response.tool_calls:
                    return response.content
                
                # 添加助手消息
                messages.append({
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["args"])
                            }
                        }
                        for tc in response.tool_calls
                    ]
                })
                
                # 执行工具调用
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    
                    console.print(f"[dim]调用工具: {tool_name}[/dim]")
                    
                    try:
                        result = await session.call_tool(tool_name, tool_args)
                        tool_result = str(result.content) if result.content else "执行成功"
                    except Exception as e:
                        tool_result = f"工具执行错误: {e}"
                    
                    # 添加工具结果
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": tool_result
                    })
            
            return "达到最大迭代次数，请简化问题"


# ==================== 记忆层级配置 ====================
MEMORY_CONFIG = {
    "recent_limit": 10,      # 短期记忆保留的对话轮数
    "medium_limit": 5,       # 中期记忆保留的摘要数
    "compress_threshold": 10, # 触发压缩的对话轮数
}

COMPRESS_PROMPT = """请将以下对话历史压缩成简洁的摘要，保留关键信息和上下文要点。
摘要应该包含：
1. 讨论的主要话题
2. 重要的结论或决定
3. 用户的关键偏好或需求

对话历史：
{conversations}

请用2-3句话输出摘要："""

MERGE_SUMMARIES_PROMPT = """请将以下多个对话摘要合并成一个更精炼的长期记忆摘要。
保留最重要的信息和模式。

摘要列表：
{summaries}

请输出合并后的精炼摘要（1-2句话）："""


# ==================== 配置管理 ====================

def load_config() -> dict:
    """加载模型配置文件"""
    if not CONFIG_FILE.exists():
        return {"models": {}, "default": None, "default_role": None}
    
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    
    return {
        "models": config.get("models", {}),
        "default": config.get("default"),
        "default_role": config.get("default_role"),
    }


def save_config(config: dict) -> None:
    """保存模型配置文件"""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)


def load_roles() -> dict:
    """加载角色配置"""
    if not ROLES_FILE.exists():
        return {}
    
    with open(ROLES_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_roles(roles: dict) -> None:
    """保存角色配置"""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(ROLES_FILE, "w", encoding="utf-8") as f:
        yaml.dump(roles, f, allow_unicode=True, default_flow_style=False)


# ==================== 记忆管理 ====================

def get_memory_file(role_name: str) -> Path:
    """获取角色记忆文件路径"""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    return MEMORY_DIR / f"{role_name}.yaml"


def load_memory(role_name: str) -> dict:
    """加载角色记忆
    
    记忆结构：
    {
        "recent": [  # 短期记忆 - 最近的完整对话
            {"role": "user", "content": "...", "timestamp": "..."},
            {"role": "assistant", "content": "...", "timestamp": "..."},
        ],
        "medium": [  # 中期记忆 - 压缩后的摘要
            {"summary": "...", "timestamp": "...", "turns": 10},
        ],
        "long": ""   # 长期记忆 - 高度压缩的整体摘要
    }
    """
    memory_file = get_memory_file(role_name)
    if not memory_file.exists():
        return {"recent": [], "medium": [], "long": ""}
    
    with open(memory_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {"recent": [], "medium": [], "long": ""}


def save_memory(role_name: str, memory: dict) -> None:
    """保存角色记忆"""
    memory_file = get_memory_file(role_name)
    with open(memory_file, "w", encoding="utf-8") as f:
        yaml.dump(memory, f, allow_unicode=True, default_flow_style=False)


def add_to_memory(role_name: str, user_msg: str, assistant_msg: str) -> None:
    """添加新对话到记忆"""
    memory = load_memory(role_name)
    timestamp = datetime.now().isoformat()
    
    memory["recent"].append({
        "role": "user",
        "content": user_msg,
        "timestamp": timestamp
    })
    memory["recent"].append({
        "role": "assistant", 
        "content": assistant_msg,
        "timestamp": timestamp
    })
    
    save_memory(role_name, memory)


def compress_memory(role_name: str, llm: ChatOpenAI) -> None:
    """压缩记忆 - 当短期记忆超过阈值时触发"""
    memory = load_memory(role_name)
    recent = memory["recent"]
    
    # 计算对话轮数（每轮包含 user + assistant）
    turns = len(recent) // 2
    
    if turns < MEMORY_CONFIG["compress_threshold"]:
        return
    
    console.print("[dim]正在压缩记忆...[/dim]")
    
    # 取出需要压缩的对话（保留最近的一半）
    keep_count = MEMORY_CONFIG["recent_limit"] * 2  # 保留的消息数
    to_compress = recent[:-keep_count] if keep_count < len(recent) else []
    
    if not to_compress:
        return
    
    # 格式化对话历史
    conv_text = "\n".join([
        f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}"
        for m in to_compress
    ])
    
    # 使用 LLM 压缩
    try:
        compress_prompt = COMPRESS_PROMPT.format(conversations=conv_text)
        response = llm.invoke([HumanMessage(content=compress_prompt)])
        summary = response.content.strip()
        
        # 添加到中期记忆
        memory["medium"].append({
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
            "turns": len(to_compress) // 2
        })
        
        # 更新短期记忆
        memory["recent"] = recent[-keep_count:] if keep_count < len(recent) else recent
        
        # 检查是否需要合并中期记忆到长期记忆
        if len(memory["medium"]) > MEMORY_CONFIG["medium_limit"]:
            merge_to_long_memory(memory, llm)
        
        save_memory(role_name, memory)
        console.print("[dim]记忆压缩完成[/dim]")
        
    except Exception as e:
        console.print(f"[yellow]记忆压缩失败: {e}[/yellow]")


def merge_to_long_memory(memory: dict, llm: ChatOpenAI) -> None:
    """合并中期记忆到长期记忆"""
    summaries = [m["summary"] for m in memory["medium"]]
    
    # 包含现有的长期记忆
    if memory["long"]:
        summaries.insert(0, f"历史背景: {memory['long']}")
    
    summaries_text = "\n".join([f"- {s}" for s in summaries])
    
    try:
        merge_prompt = MERGE_SUMMARIES_PROMPT.format(summaries=summaries_text)
        response = llm.invoke([HumanMessage(content=merge_prompt)])
        
        memory["long"] = response.content.strip()
        # 保留最近的一个中期记忆
        memory["medium"] = memory["medium"][-1:]
        
    except Exception as e:
        console.print(f"[yellow]长期记忆合并失败: {e}[/yellow]")


def build_context_messages(role_name: str, system_prompt: str) -> List:
    """构建包含记忆的上下文消息"""
    memory = load_memory(role_name)
    messages = []
    
    # 构建增强的系统提示词
    enhanced_system = system_prompt
    
    # 添加长期记忆
    if memory["long"]:
        enhanced_system += f"\n\n[长期记忆] {memory['long']}"
    
    # 添加中期记忆摘要
    if memory["medium"]:
        medium_text = " | ".join([m["summary"] for m in memory["medium"]])
        enhanced_system += f"\n\n[历史摘要] {medium_text}"
    
    messages.append(SystemMessage(content=enhanced_system))
    
    # 添加短期记忆（最近的对话）
    for msg in memory["recent"]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    
    return messages


# ==================== 模型管理 ====================

def get_model(model_name: Optional[str] = None) -> ChatOpenAI:
    """获取指定模型或默认模型"""
    config = load_config()
    
    if not config["models"]:
        console.print("[red]错误: 尚未配置任何模型，请先运行 'ask config add' 添加模型[/red]")
        sys.exit(1)
    
    name = model_name or config.get("default")
    if not name:
        name = list(config["models"].keys())[0]
    
    if name not in config["models"]:
        console.print(f"[red]错误: 模型 '{name}' 不存在[/red]")
        sys.exit(1)
    
    model_config = config["models"][name]
    
    return ChatOpenAI(
        model=model_config.get("model", "gpt-3.5-turbo"),
        openai_api_key=model_config.get("api_key"),
        openai_api_base=model_config.get("api_base"),
        temperature=model_config.get("temperature", 0.7),
    )


# ==================== 问答功能 ====================

def ask_question(
    question: str, 
    model_name: Optional[str] = None, 
    system_prompt: Optional[str] = None,
    role_name: Optional[str] = None,
    mcp_servers: Optional[List[str]] = None,
    use_tools: bool = False
):
    """向 LLM 提问并打印回答
    
    Args:
        question: 问题内容
        model_name: 模型名称
        system_prompt: 系统提示词
        role_name: 角色名称
        mcp_servers: 要使用的 MCP 服务器列表
        use_tools: 是否启用工具调用（使用默认启用的 MCP 服务器）
    """
    try:
        config = load_config()
        roles = load_roles()
        
        # 确定使用的角色
        active_role = role_name or config.get("default_role")
        role_config = roles.get(active_role, {}) if active_role else {}
        
        # 确定系统提示词
        final_system = system_prompt or role_config.get("system_prompt", "")
        
        # 确定使用的模型
        final_model = model_name or role_config.get("model") or config.get("default")
        
        llm = get_model(final_model)
        
        # 检查是否使用 MCP 工具
        use_mcp = mcp_servers is not None or use_tools
        
        if use_mcp:
            if not MCP_AVAILABLE:
                console.print("[red]错误: MCP 支持未安装，请运行: uv sync[/red]")
                sys.exit(1)
            
            # 显示状态
            status_text = f"正在思考 (模型: {final_model}, 工具模式)"
            
            with console.status(f"[bold green]{status_text}...[/bold green]"):
                response_content = asyncio.run(
                    run_with_mcp_tools(
                        question, llm, mcp_servers, final_system, active_role
                    )
                )
            
            # 输出回答
            console.print()
            console.print(Markdown(response_content))
            console.print()
            
            # 使用工具模式时暂不保存记忆
            return
        
        # 普通问答模式
        # 构建消息
        if active_role and active_role in roles:
            messages = build_context_messages(active_role, final_system)
        else:
            messages = []
            if final_system:
                messages.append(SystemMessage(content=final_system))
        
        messages.append(HumanMessage(content=question))
        
        # 显示状态
        status_text = f"正在思考"
        if final_model:
            status_text += f" (模型: {final_model}"
            if active_role:
                status_text += f", 角色: {active_role}"
            status_text += ")"
        
        with console.status(f"[bold green]{status_text}...[/bold green]"):
            response = llm.invoke(messages)
        
        # 输出回答
        console.print()
        console.print(Markdown(response.content))
        console.print()
        
        # 如果使用了角色，保存到记忆
        if active_role and active_role in roles:
            add_to_memory(active_role, question, response.content)
            # 检查是否需要压缩记忆
            compress_memory(active_role, llm)
        
    except Exception as e:
        console.print(f"[red]错误: {e}[/red]")
        sys.exit(1)


# ==================== CLI 命令 ====================

SUBCOMMANDS = ["config", "role", "mcp", "q"]


class AskCLI(click.Group):
    """自定义 CLI 组，支持直接提问"""
    
    def make_context(self, info_name, args, parent=None, **extra):
        """在创建上下文前预处理参数"""
        # 检查是否需要插入 'q' 命令
        if args:
            # 特殊处理帮助选项
            if args[0] in ['--help', '-h', 'help']:
                return super().make_context(info_name, args, parent, **extra)
            
            # 找第一个非选项参数
            first_non_option = None
            for arg in args:
                if not arg.startswith('-'):
                    first_non_option = arg
                    break
            
            # 如果第一个非选项参数不是子命令，或者全是选项（但不是帮助）
            if first_non_option is None or first_non_option not in SUBCOMMANDS:
                args = ['q'] + list(args)
        
        return super().make_context(info_name, args, parent, **extra)


@click.group(cls=AskCLI)
def cli():
    """ask.py - 终端 LLM 问答工具
    
    \b
    直接提问: ask "你的问题"
    使用角色: ask -r coder "帮我写代码"
    使用工具: ask -t "帮我查询天气"
    配置模型: ask config --help
    管理角色: ask role --help
    管理工具: ask mcp --help
    """
    pass


@cli.command("q")
@click.argument("question", nargs=-1, required=True)
@click.option("-m", "--model", help="指定使用的模型名称")
@click.option("-s", "--system", help="设置系统提示词（临时）")
@click.option("-r", "--role", help="使用指定角色")
@click.option("-t", "--tools", is_flag=True, help="启用 MCP 工具（使用默认启用的服务器）")
@click.option("--mcp", "mcp_servers", multiple=True, help="指定 MCP 服务器（可多次使用）")
def ask_cmd(question, model, system, role, tools, mcp_servers):
    """向 LLM 提问
    
    \b
    示例:
      ask "什么是机器学习?"
      ask -r coder "写一个快速排序"
      ask -t "帮我读取当前目录的文件"
      ask --mcp filesystem "列出 /tmp 目录"
    """
    servers = list(mcp_servers) if mcp_servers else None
    ask_question(" ".join(question), model, system, role, servers, tools)


# ==================== 模型配置命令 ====================

@cli.group("config")
def config_group():
    """配置模型"""
    pass


@config_group.command("add")
@click.argument("name")
@click.option("--api-base", "-b", required=True, help="API 基础 URL")
@click.option("--api-key", "-k", required=True, help="API Key")
@click.option("--model", "-m", default="gpt-3.5-turbo", help="模型名称")
@click.option("--temperature", "-t", default=0.7, type=float, help="温度参数")
@click.option("--set-default", is_flag=True, help="设置为默认模型")
def config_add(name, api_base, api_key, model, temperature, set_default):
    """添加模型配置"""
    cfg = load_config()
    
    cfg["models"][name] = {
        "api_base": api_base,
        "api_key": api_key,
        "model": model,
        "temperature": temperature,
    }
    
    if set_default or not cfg.get("default"):
        cfg["default"] = name
    
    save_config(cfg)
    console.print(f"[green]✓ 模型 '{name}' 已添加[/green]")
    if cfg["default"] == name:
        console.print(f"[blue]  已设为默认模型[/blue]")


@config_group.command("remove")
@click.argument("name")
def config_remove(name):
    """删除模型配置"""
    cfg = load_config()
    
    if name not in cfg["models"]:
        console.print(f"[red]错误: 模型 '{name}' 不存在[/red]")
        sys.exit(1)
    
    del cfg["models"][name]
    if cfg.get("default") == name:
        cfg["default"] = list(cfg["models"].keys())[0] if cfg["models"] else None
    
    save_config(cfg)
    console.print(f"[green]✓ 模型 '{name}' 已删除[/green]")


@config_group.command("default")
@click.argument("name")
def config_default(name):
    """设置默认模型"""
    cfg = load_config()
    
    if name not in cfg["models"]:
        console.print(f"[red]错误: 模型 '{name}' 不存在[/red]")
        sys.exit(1)
    
    cfg["default"] = name
    save_config(cfg)
    console.print(f"[green]✓ 默认模型已设置为 '{name}'[/green]")


@config_group.command("list")
def config_list():
    """列出所有模型配置"""
    cfg = load_config()
    
    if not cfg["models"]:
        console.print("[yellow]尚未配置任何模型[/yellow]")
        return
    
    table = Table(title="模型配置列表")
    table.add_column("名称", style="cyan")
    table.add_column("API Base", style="green")
    table.add_column("模型", style="yellow")
    table.add_column("默认", justify="center")
    
    for name, model in cfg["models"].items():
        is_default = "✓" if name == cfg.get("default") else ""
        table.add_row(name, model.get("api_base", ""), model.get("model", ""), is_default)
    
    console.print(table)


# ==================== 角色管理命令 ====================

@cli.group("role")
def role_group():
    """管理角色"""
    pass


@role_group.command("add")
@click.argument("name")
@click.option("-s", "--system", required=True, help="系统提示词")
@click.option("-m", "--model", help="绑定的模型（可选）")
@click.option("--set-default", is_flag=True, help="设置为默认角色")
def role_add(name, system, model, set_default):
    """添加新角色
    
    \b
    示例:
      ask role add coder -s "你是一个资深程序员，擅长写简洁高效的代码"
      ask role add translator -s "你是专业翻译，精通中英文互译" --set-default
    """
    roles = load_roles()
    cfg = load_config()
    
    # 如果指定了模型，检查模型是否存在
    if model and model not in cfg["models"]:
        console.print(f"[red]错误: 模型 '{model}' 不存在[/red]")
        sys.exit(1)
    
    roles[name] = {
        "system_prompt": system,
        "model": model,
        "created_at": datetime.now().isoformat(),
    }
    
    save_roles(roles)
    
    if set_default:
        cfg["default_role"] = name
        save_config(cfg)
    
    console.print(f"[green]✓ 角色 '{name}' 已创建[/green]")
    if set_default:
        console.print(f"[blue]  已设为默认角色[/blue]")


@role_group.command("remove")
@click.argument("name")
@click.option("--keep-memory", is_flag=True, help="保留记忆数据")
def role_remove(name, keep_memory):
    """删除角色"""
    roles = load_roles()
    cfg = load_config()
    
    if name not in roles:
        console.print(f"[red]错误: 角色 '{name}' 不存在[/red]")
        sys.exit(1)
    
    del roles[name]
    save_roles(roles)
    
    # 删除记忆文件
    if not keep_memory:
        memory_file = get_memory_file(name)
        if memory_file.exists():
            memory_file.unlink()
            console.print(f"[dim]记忆数据已删除[/dim]")
    
    if cfg.get("default_role") == name:
        cfg["default_role"] = None
        save_config(cfg)
    
    console.print(f"[green]✓ 角色 '{name}' 已删除[/green]")


@role_group.command("list")
def role_list():
    """列出所有角色"""
    roles = load_roles()
    cfg = load_config()
    
    if not roles:
        console.print("[yellow]尚未创建任何角色[/yellow]")
        console.print("使用 'ask role add' 创建角色")
        return
    
    table = Table(title="角色列表")
    table.add_column("名称", style="cyan")
    table.add_column("系统提示词", style="green", max_width=40)
    table.add_column("绑定模型", style="yellow")
    table.add_column("记忆轮数", justify="right")
    table.add_column("默认", justify="center")
    
    for name, role in roles.items():
        is_default = "✓" if name == cfg.get("default_role") else ""
        prompt_preview = role.get("system_prompt", "")[:37] + "..." if len(role.get("system_prompt", "")) > 40 else role.get("system_prompt", "")
        
        # 计算记忆轮数
        memory = load_memory(name)
        total_turns = len(memory["recent"]) // 2
        for m in memory["medium"]:
            total_turns += m.get("turns", 0)
        
        table.add_row(
            name,
            prompt_preview,
            role.get("model", "-"),
            str(total_turns),
            is_default
        )
    
    console.print(table)


@role_group.command("show")
@click.argument("name")
def role_show(name):
    """显示角色详情"""
    roles = load_roles()
    cfg = load_config()
    
    if name not in roles:
        console.print(f"[red]错误: 角色 '{name}' 不存在[/red]")
        sys.exit(1)
    
    role = roles[name]
    memory = load_memory(name)
    is_default = name == cfg.get("default_role")
    
    # 计算记忆统计
    recent_turns = len(memory["recent"]) // 2
    medium_count = len(memory["medium"])
    has_long = bool(memory["long"])
    
    panel_content = f"""[cyan]名称:[/cyan] {name}
[cyan]系统提示词:[/cyan]
{role.get('system_prompt', '')}

[cyan]绑定模型:[/cyan] {role.get('model', '无')}
[cyan]默认角色:[/cyan] {'是' if is_default else '否'}
[cyan]创建时间:[/cyan] {role.get('created_at', '未知')}

[cyan]记忆状态:[/cyan]
  短期记忆: {recent_turns} 轮对话
  中期记忆: {medium_count} 条摘要
  长期记忆: {'有' if has_long else '无'}"""
    
    console.print(Panel(panel_content, title=f"角色详情: {name}"))


@role_group.command("default")
@click.argument("name", required=False)
def role_default(name):
    """设置或清除默认角色"""
    cfg = load_config()
    
    if name is None:
        # 清除默认角色
        cfg["default_role"] = None
        save_config(cfg)
        console.print("[green]✓ 已清除默认角色[/green]")
        return
    
    roles = load_roles()
    if name not in roles:
        console.print(f"[red]错误: 角色 '{name}' 不存在[/red]")
        sys.exit(1)
    
    cfg["default_role"] = name
    save_config(cfg)
    console.print(f"[green]✓ 默认角色已设置为 '{name}'[/green]")


@role_group.command("clear-memory")
@click.argument("name")
@click.option("--confirm", is_flag=True, help="确认清除")
def role_clear_memory(name, confirm):
    """清除角色的记忆"""
    roles = load_roles()
    
    if name not in roles:
        console.print(f"[red]错误: 角色 '{name}' 不存在[/red]")
        sys.exit(1)
    
    if not confirm:
        console.print(f"[yellow]将清除角色 '{name}' 的所有记忆数据[/yellow]")
        console.print("添加 --confirm 参数确认操作")
        return
    
    memory_file = get_memory_file(name)
    if memory_file.exists():
        memory_file.unlink()
    
    console.print(f"[green]✓ 角色 '{name}' 的记忆已清除[/green]")


@role_group.command("memory")
@click.argument("name")
@click.option("--level", "-l", type=click.Choice(["recent", "medium", "long", "all"]), default="all", help="显示的记忆层级")
def role_memory(name, level):
    """查看角色的记忆内容"""
    roles = load_roles()
    
    if name not in roles:
        console.print(f"[red]错误: 角色 '{name}' 不存在[/red]")
        sys.exit(1)
    
    memory = load_memory(name)
    
    if level in ["recent", "all"]:
        console.print(Panel("[bold]短期记忆 (Recent)[/bold]", style="cyan"))
        if memory["recent"]:
            for msg in memory["recent"][-10:]:  # 只显示最近10条
                role_label = "[blue]用户[/blue]" if msg["role"] == "user" else "[green]助手[/green]"
                content_preview = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                console.print(f"  {role_label}: {content_preview}")
        else:
            console.print("  [dim]无[/dim]")
    
    if level in ["medium", "all"]:
        console.print(Panel("[bold]中期记忆 (Medium)[/bold]", style="yellow"))
        if memory["medium"]:
            for i, m in enumerate(memory["medium"], 1):
                console.print(f"  [{i}] {m['summary']}")
                console.print(f"      [dim]({m.get('turns', '?')} 轮对话, {m.get('timestamp', '?')})[/dim]")
        else:
            console.print("  [dim]无[/dim]")
    
    if level in ["long", "all"]:
        console.print(Panel("[bold]长期记忆 (Long)[/bold]", style="green"))
        if memory["long"]:
            console.print(f"  {memory['long']}")
        else:
            console.print("  [dim]无[/dim]")


@role_group.command("edit")
@click.argument("name")
@click.option("-s", "--system", help="新的系统提示词")
@click.option("-m", "--model", help="绑定的模型（使用 'none' 清除）")
def role_edit(name, system, model):
    """编辑角色配置"""
    roles = load_roles()
    cfg = load_config()
    
    if name not in roles:
        console.print(f"[red]错误: 角色 '{name}' 不存在[/red]")
        sys.exit(1)
    
    if system:
        roles[name]["system_prompt"] = system
        console.print("[green]✓ 系统提示词已更新[/green]")
    
    if model:
        if model.lower() == "none":
            roles[name]["model"] = None
            console.print("[green]✓ 已清除绑定模型[/green]")
        elif model not in cfg["models"]:
            console.print(f"[red]错误: 模型 '{model}' 不存在[/red]")
            sys.exit(1)
        else:
            roles[name]["model"] = model
            console.print(f"[green]✓ 已绑定模型 '{model}'[/green]")
    
    save_roles(roles)


# ==================== MCP 管理命令 ====================

@cli.group("mcp")
def mcp_group():
    """查看 MCP 服务器状态
    
    \b
    MCP 配置文件: ~/.config/ask/mcp.json
    
    编辑配置文件添加/修改 MCP 服务器
    """
    pass


@mcp_group.command("list")
def mcp_list():
    """列出所有 MCP 服务器"""
    cfg = load_mcp_config()
    servers = cfg.get("mcpServers", {})
    enabled = cfg.get("enabled")
    
    if not servers:
        console.print("[yellow]尚未配置任何 MCP 服务器[/yellow]")
        console.print(f"\n编辑配置文件: [cyan]{MCP_FILE}[/cyan]")
        console.print("\n示例配置:")
        console.print('''[dim]{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    }
  }
}[/dim]''')
        return
    
    table = Table(title="MCP 服务器列表")
    table.add_column("名称", style="cyan")
    table.add_column("命令", style="green")
    table.add_column("启用", justify="center")
    
    # 如果 enabled 为 None，表示全部启用
    all_enabled = enabled is None
    enabled_set = set(enabled) if enabled else set()
    
    for name, server in servers.items():
        is_enabled = "✓" if all_enabled or name in enabled_set else ""
        cmd = server.get("command", "")
        args = " ".join(server.get("args", []))
        cmd_str = f"{cmd} {args}".strip()
        if len(cmd_str) > 50:
            cmd_str = cmd_str[:47] + "..."
        table.add_row(name, cmd_str, is_enabled)
    
    console.print(table)
    console.print(f"\n配置文件: [cyan]{MCP_FILE}[/cyan]")


@mcp_group.command("tools")
@click.argument("name", required=False)
def mcp_tools(name):
    """列出 MCP 服务器提供的工具"""
    if not MCP_AVAILABLE:
        console.print("[red]错误: MCP 支持未安装，请运行: uv sync[/red]")
        sys.exit(1)
    
    cfg = load_mcp_config()
    servers = cfg.get("mcpServers", {})
    
    if not servers:
        console.print("[yellow]尚未配置任何 MCP 服务器[/yellow]")
        return
    
    if name:
        if name not in servers:
            console.print(f"[red]错误: MCP 服务器 '{name}' 不存在[/red]")
            sys.exit(1)
        server_name = name
    else:
        available = get_available_mcp_servers()
        if not available:
            console.print("[yellow]没有可用的 MCP 服务器[/yellow]")
            return
        server_name = available[0]
    
    server_config = servers.get(server_name)
    
    async def list_tools():
        command = server_config.get("command")
        args = server_config.get("args", [])
        env = server_config.get("env")
        
        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                return result.tools
    
    with console.status(f"[bold green]正在连接 MCP 服务器 '{server_name}'...[/bold green]"):
        try:
            tools = asyncio.run(list_tools())
        except Exception as e:
            console.print(f"[red]错误: {e}[/red]")
            sys.exit(1)
    
    if not tools:
        console.print("[yellow]服务器没有提供任何工具[/yellow]")
        return
    
    table = Table(title=f"MCP 工具 ({server_name})")
    table.add_column("工具名称", style="cyan")
    table.add_column("描述", style="green")
    
    for tool in tools:
        desc = tool.description or ""
        if len(desc) > 60:
            desc = desc[:57] + "..."
        table.add_row(tool.name, desc)
    
    console.print(table)
    console.print(f"\n共 {len(tools)} 个工具")


@mcp_group.command("path")
def mcp_path():
    """显示 MCP 配置文件路径"""
    console.print(f"配置文件: [cyan]{MCP_FILE}[/cyan]")
    if MCP_FILE.exists():
        console.print("[green]文件已存在[/green]")
    else:
        console.print("[yellow]文件不存在，请创建[/yellow]")


if __name__ == "__main__":
    cli()
