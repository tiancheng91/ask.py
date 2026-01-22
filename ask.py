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
from functools import lru_cache
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

# i18n
from i18n import t, set_lang, get_lang, SUPPORTED_LANGS

# 配置文件路径
CONFIG_DIR = Path.home() / ".config" / "ask"
CONFIG_FILE = CONFIG_DIR / "config.yaml"
ROLES_FILE = CONFIG_DIR / "roles.yaml"
MEMORY_DIR = CONFIG_DIR / "memory"
MCP_FILE = CONFIG_DIR / "mcp.json"

console = Console()

# MCP 可用性标志（延迟导入）
MCP_AVAILABLE = None  # None 表示尚未检测
_mcp_module = None
_mcp_stdio_module = None


def _lazy_import_mcp():
    """延迟导入 MCP 模块，只在实际使用时加载"""
    global MCP_AVAILABLE, _mcp_module, _mcp_stdio_module
    
    if MCP_AVAILABLE is not None:
        return MCP_AVAILABLE
    
    try:
        import mcp
        import mcp.client.stdio
        _mcp_module = mcp
        _mcp_stdio_module = mcp.client.stdio
        MCP_AVAILABLE = True
    except ImportError:
        MCP_AVAILABLE = False
    
    return MCP_AVAILABLE


def _get_mcp_classes():
    """获取 MCP 相关类，需要先调用 _lazy_import_mcp()"""
    if not MCP_AVAILABLE:
        return None, None, None
    return (
        _mcp_module.ClientSession,
        _mcp_module.StdioServerParameters,
        _mcp_stdio_module.stdio_client
    )

# ==================== MCP 管理 ====================

import shutil

# 缓存包运行器检测结果，避免重复检测
_PACKAGE_RUNNER_CACHE = None

def detect_package_runner() -> str:
    """检测可用的包运行器（uvx 或 pipx），结果会被缓存
    
    Returns:
        "uvx" 或 "pipx"，优先使用 uvx
    """
    global _PACKAGE_RUNNER_CACHE
    
    if _PACKAGE_RUNNER_CACHE is not None:
        return _PACKAGE_RUNNER_CACHE
    
    if shutil.which("uvx"):
        _PACKAGE_RUNNER_CACHE = "uvx"
    elif shutil.which("pipx"):
        _PACKAGE_RUNNER_CACHE = "pipx"
    else:
        # 默认返回 uvx，让用户自行安装
        _PACKAGE_RUNNER_CACHE = "uvx"
    
    return _PACKAGE_RUNNER_CACHE


def get_default_mcp_config() -> dict:
    """生成默认 MCP 配置，根据系统环境选择 uvx 或 pipx
    
    Returns:
        默认 MCP 配置字典
    """
    runner = detect_package_runner()
    return {
        "mcpServers": {
            "time": {
                "command": runner,
                "args": ["mcp-server-time"]
            },
            "shell": {
                "command": runner,
                "args": ["mcp-shell-server"],
                "env": {
                    "ALLOW_COMMANDS": "ls,cat,head,tail,find,grep,wc,pwd,echo,mkdir,cp,mv,touch,date,whoami,hostname,ps,du"
                }
            }
        },
        "enabled": ["time", "shell"]
    }


@lru_cache(maxsize=1)
@lru_cache(maxsize=1)
def load_mcp_config() -> dict:
    """加载 MCP 服务器配置 (JSON 格式)，结果会被缓存
    
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
        # 首次使用时创建默认配置（自动检测 uvx/pipx）
        default_config = get_default_mcp_config()
        save_mcp_config(default_config)
        return default_config.copy()
    
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


class MCPConnection:
    """MCP 连接管理器，支持连接复用"""
    
    def __init__(self, server_name: str, server_config: dict):
        self.server_name = server_name
        self.server_config = server_config
        self.session = None
        self.tools = []
        self._context_stack = []
    
    async def connect(self):
        """建立连接"""
        if self.server_config.get("type") == "sse":
            raise RuntimeError("SSE 类型暂不支持，请使用 stdio 类型")
        
        ClientSession, StdioServerParameters, stdio_client = _get_mcp_classes()
        
        command = self.server_config.get("command")
        args = self.server_config.get("args", [])
        env = self.server_config.get("env")
        
        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env
        )
        
        # 进入 stdio_client 上下文
        stdio_ctx = stdio_client(server_params)
        read, write = await stdio_ctx.__aenter__()
        self._context_stack.append(stdio_ctx)
        
        # 进入 ClientSession 上下文
        session_ctx = ClientSession(read, write)
        self.session = await session_ctx.__aenter__()
        self._context_stack.append(session_ctx)
        
        await self.session.initialize()
        
        # 获取工具列表
        tools_result = await self.session.list_tools()
        self.tools = tools_result.tools
        
        return self.tools
    
    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """调用工具（复用已建立的连接）"""
        if not self.session:
            raise RuntimeError(f"MCP 连接未建立: {self.server_name}")
        
        result = await self.session.call_tool(tool_name, arguments)
        return result
    
    async def close(self):
        """关闭连接"""
        # 按逆序退出上下文
        for ctx in reversed(self._context_stack):
            try:
                await ctx.__aexit__(None, None, None)
            except Exception:
                pass
        self._context_stack.clear()
        self.session = None


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


async def _connect_single_server(server_name: str) -> Optional[MCPConnection]:
    """连接单个 MCP 服务器（用于并行连接）"""
    server_config = get_mcp_server_by_name(server_name)
    if not server_config:
        return None
    
    conn = MCPConnection(server_name, server_config)
    try:
        await conn.connect()
        return conn
    except Exception as e:
        console.print(f"[yellow]{t('error.mcp_connect_failed', name=server_name, error=str(e))}[/yellow]")
        return None


async def connect_mcp_servers(servers_to_use: List[str]) -> tuple:
    """并行连接多个 MCP 服务器
    
    Returns:
        (connections, all_tools, tool_to_connection): 
        连接列表、所有工具列表、工具名到连接的映射
    """
    # 并行连接所有服务器
    tasks = [_connect_single_server(name) for name in servers_to_use]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    connections = []
    all_tools = []
    tool_to_connection = {}  # tool_name -> MCPConnection
    
    for result in results:
        if isinstance(result, Exception):
            continue
        if result is None:
            continue
        
        conn = result
        connections.append(conn)
        
        for tool in conn.tools:
            all_tools.append(tool)
            tool_to_connection[tool.name] = conn
    
    return connections, all_tools, tool_to_connection


async def close_mcp_connections(connections: List[MCPConnection]) -> None:
    """关闭所有 MCP 连接"""
    for conn in connections:
        try:
            await conn.close()
        except Exception:
            pass


async def run_with_mcp_tools(
    question: str,
    llm: ChatOpenAI,
    server_names: Optional[List[str]] = None,
    system_prompt: Optional[str] = None,
    role_name: Optional[str] = None
) -> str:
    """使用 MCP 工具执行问答（ReAct 模式，支持多服务器，连接复用）"""
    if not _lazy_import_mcp():
        raise RuntimeError(t("error.mcp_not_installed"))
    
    # 确定要使用的服务器
    if server_names:
        servers_to_use = server_names
    else:
        servers_to_use = get_available_mcp_servers(role_name)
    
    if not servers_to_use:
        raise RuntimeError(t("error.no_mcp_servers"))
    
    # 并行连接所有服务器（连接复用：整个会话中保持连接）
    connections, all_tools, tool_to_connection = await connect_mcp_servers(servers_to_use)
    
    try:
        if not all_tools:
            raise RuntimeError(t("error.no_mcp_tools"))
        
        # 转换为 OpenAI 格式
        openai_tools = convert_mcp_tools_to_openai(all_tools)
        
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
            
            # 执行工具调用（复用已建立的连接）
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                console.print(f"[dim]{t('status.tool_call', name=tool_name)}[/dim]")
                
                # 找到工具对应的连接
                conn = tool_to_connection.get(tool_name)
                if not conn:
                    tool_result = t("error.tool_not_found", name=tool_name)
                else:
                    try:
                        result = await conn.call_tool(tool_name, tool_args)
                        tool_result = str(result.content) if result.content else t("status.tool_success")
                    except Exception as e:
                        tool_result = t("error.tool_error", error=str(e))
                
                # 添加工具结果
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": tool_result
                })
        
        return t("error.max_iterations")
    
    finally:
        # 确保关闭所有连接
        await close_mcp_connections(connections)


# ==================== 记忆层级配置 ====================
MEMORY_CONFIG = {
    "recent_limit": 10,      # 短期记忆保留的对话轮数
    "medium_limit": 5,       # 中期记忆保留的摘要数
    "compress_threshold": 10, # 触发压缩的对话轮数
}

COMPRESS_PROMPT = """Please compress the following conversation history into a concise summary, keeping key information and context.
Summary should include:
1. Main topics discussed
2. Important conclusions or decisions
3. User's key preferences or needs

Conversation history:
{conversations}

Please output summary in 2-3 sentences:"""

MERGE_SUMMARIES_PROMPT = """Please merge the following summaries into a more refined long-term memory summary.
Keep the most important information and patterns.

Summary list:
{summaries}

Please output merged refined summary (1-2 sentences):"""


# ==================== 配置管理 ====================

@lru_cache(maxsize=1)
def load_config() -> dict:
    """加载模型配置文件，结果会被缓存"""
    if not CONFIG_FILE.exists():
        return {"models": {}, "default": None, "default_role": None, "lang": None}
    
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    
    # 设置语言
    lang = config.get("lang")
    if lang and lang in SUPPORTED_LANGS:
        set_lang(lang)
    
    return {
        "models": config.get("models", {}),
        "default": config.get("default"),
        "default_role": config.get("default_role"),
        "lang": lang,
    }


def save_config(config: dict) -> None:
    """保存模型配置文件"""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)


@lru_cache(maxsize=1)
def load_roles() -> dict:
    """加载角色配置，结果会被缓存"""
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
    
    console.print(f"[dim]{t('status.compressing')}[/dim]")
    
    # 取出需要压缩的对话（保留最近的一半）
    keep_count = MEMORY_CONFIG["recent_limit"] * 2  # 保留的消息数
    to_compress = recent[:-keep_count] if keep_count < len(recent) else []
    
    if not to_compress:
        return
    
    # 格式化对话历史
    conv_text = "\n".join([
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
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
        console.print(f"[dim]{t('status.compressed')}[/dim]")
        
    except Exception as e:
        console.print(f"[yellow]{t('error.general', error=str(e))}[/yellow]")


def merge_to_long_memory(memory: dict, llm: ChatOpenAI) -> None:
    """合并中期记忆到长期记忆"""
    summaries = [m["summary"] for m in memory["medium"]]
    
    # 包含现有的长期记忆
    if memory["long"]:
        summaries.insert(0, f"Historical background: {memory['long']}")
    
    summaries_text = "\n".join([f"- {s}" for s in summaries])
    
    try:
        merge_prompt = MERGE_SUMMARIES_PROMPT.format(summaries=summaries_text)
        response = llm.invoke([HumanMessage(content=merge_prompt)])
        
        memory["long"] = response.content.strip()
        # 保留最近的一个中期记忆
        memory["medium"] = memory["medium"][-1:]
        
    except Exception as e:
        console.print(f"[yellow]{t('error.general', error=str(e))}[/yellow]")


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
        console.print(f"[red]{t('error.no_model_config')}[/red]")
        sys.exit(1)
    
    name = model_name or config.get("default")
    if not name:
        name = list(config["models"].keys())[0]
    
    if name not in config["models"]:
        console.print(f"[red]{t('error.model_not_found', name=name)}[/red]")
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
            if not _lazy_import_mcp():
                console.print(f"[red]{t('error.mcp_not_installed')}[/red]")
                sys.exit(1)
            
            # 显示状态
            status_text = t("status.thinking_tools", model=final_model)
            
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
        if final_model and active_role:
            status_text = t("status.thinking_with_role", model=final_model, role=active_role)
        elif final_model:
            status_text = t("status.thinking_with_model", model=final_model)
        else:
            status_text = t("status.thinking")
        
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
        console.print(f"[red]{t('error.general', error=str(e))}[/red]")
        sys.exit(1)


# ==================== CLI 命令 ====================

SUBCOMMANDS = ["model", "role", "mcp", "q"]


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
    """ask.py - Terminal LLM Q&A Tool / 终端 LLM 问答工具
    
    \b
    Quick ask / 快速提问: ask "your question"
    Use role / 使用角色: ask -r coder "help me code"
    Use tools / 使用工具: ask -t "check the weather"
    Use shell / 使用 Shell: ask --mcp shell "list files"
    Manage models / 管理模型: ask model --help
    Manage roles / 管理角色: ask role --help
    Manage tools / 管理工具: ask mcp --help
    """
    # 初始化时加载配置以设置语言
    load_config()


@cli.command("q")
@click.argument("question", nargs=-1, required=True)
@click.option("-m", "--model", help="Specify model name")
@click.option("-s", "--system", help="Set system prompt (temporary)")
@click.option("-r", "--role", help="Use specified role")
@click.option("-t", "--tools", is_flag=True, help="Enable MCP tools")
@click.option("--mcp", "mcp_servers", multiple=True, help="Specify MCP server (can be used multiple times)")
def ask_cmd(question, model, system, role, tools, mcp_servers):
    """Ask LLM a question / 向 LLM 提问
    
    \b
    Examples / 示例:
      ask "What is machine learning?"
      ask -r coder "Write a quicksort"
      ask -t "What time is it?"
      ask --mcp shell "List files in current directory"
    """
    servers = list(mcp_servers) if mcp_servers else None
    ask_question(" ".join(question), model, system, role, servers, tools)


# ==================== 模型配置命令 ====================

@cli.group("model")
def model_group():
    """管理模型"""
    pass


@model_group.command("add")
@click.argument("name")
@click.option("--api-base", "-b", required=True, help="API base URL")
@click.option("--api-key", "-k", required=True, help="API Key")
@click.option("--model", "-m", default="gpt-3.5-turbo", help="Model name")
@click.option("--temperature", "-t", default=0.7, type=float, help="Temperature")
@click.option("--set-default", is_flag=True, help="Set as default")
def config_add(name, api_base, api_key, model, temperature, set_default):
    """Add model configuration / 添加模型配置"""
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
    console.print(f"[green]✓ {t('success.model_added', name=name)}[/green]")
    if cfg["default"] == name:
        console.print(f"[blue]  {t('success.set_as_default')}[/blue]")


@model_group.command("remove")
@click.argument("name")
def config_remove(name):
    """Remove model configuration / 删除模型配置"""
    cfg = load_config()
    
    if name not in cfg["models"]:
        console.print(f"[red]{t('error.model_not_found', name=name)}[/red]")
        sys.exit(1)
    
    del cfg["models"][name]
    if cfg.get("default") == name:
        cfg["default"] = list(cfg["models"].keys())[0] if cfg["models"] else None
    
    save_config(cfg)
    console.print(f"[green]✓ {t('success.model_removed', name=name)}[/green]")


@model_group.command("default")
@click.argument("name")
def config_default(name):
    """Set default model / 设置默认模型"""
    cfg = load_config()
    
    if name not in cfg["models"]:
        console.print(f"[red]{t('error.model_not_found', name=name)}[/red]")
        sys.exit(1)
    
    cfg["default"] = name
    save_config(cfg)
    console.print(f"[green]✓ {t('success.model_default', name=name)}[/green]")


@model_group.command("list")
def config_list():
    """List all model configurations / 列出所有模型配置"""
    cfg = load_config()
    
    if not cfg["models"]:
        console.print(f"[yellow]{t('hint.no_model')}[/yellow]")
        return
    
    table = Table(title=t("table.model_list"))
    table.add_column(t("table.col.name"), style="cyan")
    table.add_column(t("table.col.api_base"), style="green")
    table.add_column(t("table.col.model"), style="yellow")
    table.add_column(t("table.col.default"), justify="center")
    
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
@click.option("-s", "--system", required=True, help="System prompt")
@click.option("-m", "--model", help="Bind model (optional)")
@click.option("--set-default", is_flag=True, help="Set as default")
def role_add(name, system, model, set_default):
    """Add new role / 添加新角色"""
    roles = load_roles()
    cfg = load_config()
    
    # 如果指定了模型，检查模型是否存在
    if model and model not in cfg["models"]:
        console.print(f"[red]{t('error.model_not_found', name=model)}[/red]")
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
    
    console.print(f"[green]✓ {t('success.role_created', name=name)}[/green]")
    if set_default:
        console.print(f"[blue]  {t('hint.set_as_default_role')}[/blue]")


@role_group.command("remove")
@click.argument("name")
@click.option("--keep-memory", is_flag=True, help="Keep memory data")
def role_remove(name, keep_memory):
    """Remove role / 删除角色"""
    roles = load_roles()
    cfg = load_config()
    
    if name not in roles:
        console.print(f"[red]{t('error.role_not_found', name=name)}[/red]")
        sys.exit(1)
    
    del roles[name]
    save_roles(roles)
    
    # 删除记忆文件
    if not keep_memory:
        memory_file = get_memory_file(name)
        if memory_file.exists():
            memory_file.unlink()
            console.print(f"[dim]{t('success.memory_deleted')}[/dim]")
    
    if cfg.get("default_role") == name:
        cfg["default_role"] = None
        save_config(cfg)
    
    console.print(f"[green]✓ {t('success.role_removed', name=name)}[/green]")


@role_group.command("list")
def role_list():
    """List all roles / 列出所有角色"""
    roles = load_roles()
    cfg = load_config()
    
    if not roles:
        console.print(f"[yellow]{t('hint.no_role')}[/yellow]")
        console.print(t('hint.create_role'))
        return
    
    table = Table(title=t("table.role_list"))
    table.add_column(t("table.col.name"), style="cyan")
    table.add_column(t("table.col.system_prompt"), style="green", max_width=40)
    table.add_column(t("table.col.bind_model"), style="yellow")
    table.add_column(t("table.col.memory_turns"), justify="right")
    table.add_column(t("table.col.default"), justify="center")
    
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
    """Show role details / 显示角色详情"""
    roles = load_roles()
    cfg = load_config()
    
    if name not in roles:
        console.print(f"[red]{t('error.role_not_found', name=name)}[/red]")
        sys.exit(1)
    
    role = roles[name]
    memory = load_memory(name)
    is_default = name == cfg.get("default_role")
    
    # 计算记忆统计
    recent_turns = len(memory["recent"]) // 2
    medium_count = len(memory["medium"])
    has_long = bool(memory["long"])
    
    panel_content = f"""[cyan]{t('detail.name')}[/cyan] {name}
[cyan]{t('detail.system_prompt')}[/cyan]
{role.get('system_prompt', '')}

[cyan]{t('detail.bind_model')}[/cyan] {role.get('model') or t('detail.none')}
[cyan]{t('detail.is_default')}[/cyan] {t('detail.yes') if is_default else t('detail.no')}
[cyan]{t('detail.created_at')}[/cyan] {role.get('created_at', t('detail.unknown'))}

[cyan]{t('detail.memory_status')}[/cyan]
  {t('detail.recent_turns', count=recent_turns)}
  {t('detail.medium_count', count=medium_count)}
  {t('detail.long_status')} {t('detail.has') if has_long else t('detail.no_data')}"""
    
    console.print(Panel(panel_content, title=t("panel.role_detail", name=name)))


@role_group.command("default")
@click.argument("name", required=False)
def role_default(name):
    """Set or clear default role / 设置或清除默认角色"""
    cfg = load_config()
    
    if name is None:
        # 清除默认角色
        cfg["default_role"] = None
        save_config(cfg)
        console.print(f"[green]✓ {t('success.role_default_cleared')}[/green]")
        return
    
    roles = load_roles()
    if name not in roles:
        console.print(f"[red]{t('error.role_not_found', name=name)}[/red]")
        sys.exit(1)
    
    cfg["default_role"] = name
    save_config(cfg)
    console.print(f"[green]✓ {t('success.role_default_set', name=name)}[/green]")


@role_group.command("clear-memory")
@click.argument("name")
@click.option("--confirm", is_flag=True, help="Confirm operation")
def role_clear_memory(name, confirm):
    """Clear role memory / 清除角色记忆"""
    roles = load_roles()
    
    if name not in roles:
        console.print(f"[red]{t('error.role_not_found', name=name)}[/red]")
        sys.exit(1)
    
    if not confirm:
        console.print(f"[yellow]{t('hint.confirm_clear', name=name)}[/yellow]")
        console.print(t('hint.add_confirm'))
        return
    
    memory_file = get_memory_file(name)
    if memory_file.exists():
        memory_file.unlink()
    
    console.print(f"[green]✓ {t('success.memory_cleared', name=name)}[/green]")


@role_group.command("memory")
@click.argument("name")
@click.option("--level", "-l", type=click.Choice(["recent", "medium", "long", "all"]), default="all", help="Memory level to display")
def role_memory(name, level):
    """View role memory / 查看角色记忆"""
    roles = load_roles()
    
    if name not in roles:
        console.print(f"[red]{t('error.role_not_found', name=name)}[/red]")
        sys.exit(1)
    
    memory = load_memory(name)
    
    if level in ["recent", "all"]:
        console.print(Panel(f"[bold]{t('panel.recent_memory')}[/bold]", style="cyan"))
        if memory["recent"]:
            for msg in memory["recent"][-10:]:  # 只显示最近10条
                role_label = f"[blue]{t('memory.user')}[/blue]" if msg["role"] == "user" else f"[green]{t('memory.assistant')}[/green]"
                content_preview = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                console.print(f"  {role_label}: {content_preview}")
        else:
            console.print(f"  [dim]{t('detail.no_data')}[/dim]")
    
    if level in ["medium", "all"]:
        console.print(Panel(f"[bold]{t('panel.medium_memory')}[/bold]", style="yellow"))
        if memory["medium"]:
            for i, m in enumerate(memory["medium"], 1):
                console.print(f"  [{i}] {m['summary']}")
                console.print(f"      [dim]{t('memory.turns_info', turns=m.get('turns', '?'), time=m.get('timestamp', '?'))}[/dim]")
        else:
            console.print(f"  [dim]{t('detail.no_data')}[/dim]")
    
    if level in ["long", "all"]:
        console.print(Panel(f"[bold]{t('panel.long_memory')}[/bold]", style="green"))
        if memory["long"]:
            console.print(f"  {memory['long']}")
        else:
            console.print(f"  [dim]{t('detail.no_data')}[/dim]")


@role_group.command("edit")
@click.argument("name")
@click.option("-s", "--system", help="New system prompt")
@click.option("-m", "--model", help="Bind model (use 'none' to clear)")
def role_edit(name, system, model):
    """Edit role configuration / 编辑角色配置"""
    roles = load_roles()
    cfg = load_config()
    
    if name not in roles:
        console.print(f"[red]{t('error.role_not_found', name=name)}[/red]")
        sys.exit(1)
    
    if system:
        roles[name]["system_prompt"] = system
        console.print(f"[green]✓ {t('success.system_updated')}[/green]")
    
    if model:
        if model.lower() == "none":
            roles[name]["model"] = None
            console.print(f"[green]✓ {t('success.model_unbound')}[/green]")
        elif model not in cfg["models"]:
            console.print(f"[red]{t('error.model_not_found', name=model)}[/red]")
            sys.exit(1)
        else:
            roles[name]["model"] = model
            console.print(f"[green]✓ {t('success.model_bound', name=model)}[/green]")
    
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
    """List all MCP servers / 列出所有 MCP 服务器"""
    cfg = load_mcp_config()
    servers = cfg.get("mcpServers", {})
    enabled = cfg.get("enabled")
    
    if not servers:
        console.print(f"[yellow]{t('hint.no_mcp')}[/yellow]")
        console.print(f"\n{t('hint.edit_config', path=str(MCP_FILE))}")
        console.print(f"\n{t('hint.example_config')}")
        console.print('''[dim]{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    }
  }
}[/dim]''')
        return
    
    table = Table(title=t("table.mcp_list"))
    table.add_column(t("table.col.name"), style="cyan")
    table.add_column(t("table.col.command"), style="green")
    table.add_column(t("table.col.enabled"), justify="center")
    
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
    console.print(f"\n{t('hint.config_file', path=str(MCP_FILE))}")


@mcp_group.command("tools")
@click.argument("name", required=False)
def mcp_tools(name):
    """List MCP server tools / 列出 MCP 服务器提供的工具"""
    if not _lazy_import_mcp():
        console.print(f"[red]{t('error.mcp_not_installed')}[/red]")
        sys.exit(1)
    
    cfg = load_mcp_config()
    servers = cfg.get("mcpServers", {})
    
    if not servers:
        console.print(f"[yellow]{t('hint.no_mcp')}[/yellow]")
        return
    
    if name:
        if name not in servers:
            console.print(f"[red]{t('error.mcp_server_not_found', name=name)}[/red]")
            sys.exit(1)
        server_name = name
    else:
        available = get_available_mcp_servers()
        if not available:
            console.print(f"[yellow]{t('hint.no_available_mcp')}[/yellow]")
            return
        server_name = available[0]
    
    server_config = servers.get(server_name)
    
    async def list_tools():
        conn = MCPConnection(server_name, server_config)
        try:
            await conn.connect()
            return conn.tools
        finally:
            await conn.close()
    
    with console.status(f"[bold green]{t('status.connecting_mcp', name=server_name)}[/bold green]"):
        try:
            tools = asyncio.run(list_tools())
        except Exception as e:
            console.print(f"[red]{t('error.general', error=str(e))}[/red]")
            sys.exit(1)
    
    if not tools:
        console.print(f"[yellow]{t('error.no_mcp_tools')}[/yellow]")
        return
    
    table = Table(title=t("table.mcp_tools", name=server_name))
    table.add_column(t("table.col.tool_name"), style="cyan")
    table.add_column(t("table.col.description"), style="green")
    
    for tool in tools:
        desc = tool.description or ""
        if len(desc) > 60:
            desc = desc[:57] + "..."
        table.add_row(tool.name, desc)
    
    console.print(table)
    console.print(f"\n{t('hint.total_tools', count=len(tools))}")


@mcp_group.command("path")
def mcp_path():
    """Show MCP config file path / 显示 MCP 配置文件路径"""
    console.print(f"{t('hint.config_file', path=str(MCP_FILE))}")
    if MCP_FILE.exists():
        console.print(f"[green]{t('hint.file_exists')}[/green]")
    else:
        console.print(f"[yellow]{t('hint.file_not_exists')}[/yellow]")


if __name__ == "__main__":
    cli()
