"""
i18n.py - 多语言支持模块

支持语言：
- en: English
- zh-cn: 简体中文
- zh-tw: 繁體中文
- ja: 日本語

使用方法：
    from i18n import t, set_lang, get_lang
    
    # 设置语言
    set_lang("zh-cn")
    
    # 获取翻译
    print(t("error.model_not_found", name="gpt-4"))
"""

import os
from typing import Optional, Dict, Any

# 支持的语言
SUPPORTED_LANGS = ["en", "zh-cn", "zh-tw", "ja"]
DEFAULT_LANG = "en"

# 当前语言
_current_lang: Optional[str] = None

# ==================== 翻译字典 ====================

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "en": {
        # CLI 帮助
        "cli.description": "ask.py - Terminal LLM Q&A Tool",
        "cli.quick_ask": "Quick ask: ask \"your question\"",
        "cli.use_role": "Use role: ask -r coder \"help me code\"",
        "cli.use_tools": "Use tools: ask -t \"check the weather\"",
        "cli.manage_model": "Manage models: ask model --help",
        "cli.manage_role": "Manage roles: ask role --help",
        "cli.manage_mcp": "Manage tools: ask mcp --help",
        
        # 命令描述
        "cmd.q.desc": "Ask LLM a question",
        "cmd.q.example": """Examples:
  ask "What is machine learning?"
  ask -r coder "Write a quicksort"
  ask -t "Read files in current directory"
  ask --mcp filesystem "List /tmp directory\"""",
        "cmd.model.desc": "Manage models",
        "cmd.model.add.desc": "Add model configuration",
        "cmd.model.remove.desc": "Remove model configuration",
        "cmd.model.default.desc": "Set default model",
        "cmd.model.list.desc": "List all model configurations",
        "cmd.role.desc": "Manage roles",
        "cmd.role.add.desc": "Add new role",
        "cmd.role.add.example": """Examples:
  ask role add coder -s "You are a senior programmer"
  ask role add translator -s "You are a professional translator" --set-default""",
        "cmd.role.remove.desc": "Remove role",
        "cmd.role.list.desc": "List all roles",
        "cmd.role.show.desc": "Show role details",
        "cmd.role.default.desc": "Set or clear default role",
        "cmd.role.edit.desc": "Edit role configuration",
        "cmd.role.memory.desc": "View role memory",
        "cmd.role.clear_memory.desc": "Clear role memory",
        "cmd.mcp.desc": "View MCP server status",
        "cmd.mcp.config_hint": "MCP config file: ~/.config/ask/mcp.json\n\nEdit config file to add/modify MCP servers",
        "cmd.mcp.list.desc": "List all MCP servers",
        "cmd.mcp.tools.desc": "List tools provided by MCP server",
        "cmd.mcp.path.desc": "Show MCP config file path",
        
        # 选项
        "opt.model": "Specify model name",
        "opt.system": "Set system prompt (temporary)",
        "opt.role": "Use specified role",
        "opt.tools": "Enable MCP tools (use default enabled servers)",
        "opt.mcp": "Specify MCP server (can be used multiple times)",
        "opt.api_base": "API base URL",
        "opt.api_key": "API Key",
        "opt.model_name": "Model name",
        "opt.temperature": "Temperature parameter",
        "opt.set_default": "Set as default",
        "opt.system_prompt": "System prompt",
        "opt.bind_model": "Bind model (optional)",
        "opt.keep_memory": "Keep memory data",
        "opt.confirm": "Confirm operation",
        "opt.level": "Memory level to display",
        "opt.new_system": "New system prompt",
        "opt.new_model": "Bind model (use 'none' to clear)",
        
        # 表格标题
        "table.model_list": "Model Configuration List",
        "table.col.name": "Name",
        "table.col.api_base": "API Base",
        "table.col.model": "Model",
        "table.col.default": "Default",
        "table.role_list": "Role List",
        "table.col.system_prompt": "System Prompt",
        "table.col.bind_model": "Bound Model",
        "table.col.memory_turns": "Memory Turns",
        "table.mcp_list": "MCP Server List",
        "table.col.command": "Command",
        "table.col.enabled": "Enabled",
        "table.mcp_tools": "MCP Tools ({name})",
        "table.col.tool_name": "Tool Name",
        "table.col.description": "Description",
        
        # 成功消息
        "success.model_added": "Model '{name}' added",
        "success.model_removed": "Model '{name}' removed",
        "success.model_default": "Default model set to '{name}'",
        "success.set_as_default": "Set as default",
        "success.role_created": "Role '{name}' created",
        "success.role_removed": "Role '{name}' removed",
        "success.role_default_set": "Default role set to '{name}'",
        "success.role_default_cleared": "Default role cleared",
        "success.memory_cleared": "Role '{name}' memory cleared",
        "success.memory_deleted": "Memory data deleted",
        "success.system_updated": "System prompt updated",
        "success.model_bound": "Bound to model '{name}'",
        "success.model_unbound": "Model binding cleared",
        
        # 错误消息
        "error.no_model_config": "No model configured, please run 'ask model add' first",
        "error.model_not_found": "Model '{name}' not found",
        "error.role_not_found": "Role '{name}' not found",
        "error.mcp_not_installed": "MCP support not installed, please run: uv sync",
        "error.mcp_server_not_found": "MCP server '{name}' not found",
        "error.mcp_connect_failed": "Failed to connect MCP server '{name}': {error}",
        "error.no_mcp_servers": "No MCP servers available, please configure ~/.config/ask/mcp.json",
        "error.no_mcp_tools": "MCP server provides no tools",
        "error.tool_not_found": "Tool '{name}' not found in any server",
        "error.max_iterations": "Max iterations reached, please simplify your question",
        "error.tool_error": "Tool execution error: {error}",
        "error.general": "Error: {error}",
        
        # 状态消息
        "status.thinking": "Thinking",
        "status.thinking_with_model": "Thinking (model: {model})",
        "status.thinking_with_role": "Thinking (model: {model}, role: {role})",
        "status.thinking_tools": "Thinking (model: {model}, tools mode)",
        "status.connecting_mcp": "Connecting to MCP server '{name}'...",
        "status.compressing": "Compressing memory...",
        "status.compressed": "Memory compressed",
        "status.tool_call": "Calling tool: {name}",
        "status.tool_success": "Executed successfully",
        "status.shell_command": "Shell command: {cmd}",
        "status.shell_confirm": "Execute this command? [y/N]: ",
        "status.shell_cancelled": "Command cancelled",
        "status.shell_output": "Output:",
        
        # Shell 相关
        "error.shell_not_installed": "Shell tool not installed, please run: uv sync",
        "error.shell_rejected": "Command rejected by user",
        
        # 提示消息
        "hint.no_model": "No model configured",
        "hint.no_role": "No role created",
        "hint.create_role": "Use 'ask role add' to create a role",
        "hint.no_mcp": "No MCP server configured",
        "hint.edit_config": "Edit config file: {path}",
        "hint.example_config": "Example config:",
        "hint.confirm_clear": "Will clear all memory for role '{name}'",
        "hint.add_confirm": "Add --confirm to confirm",
        "hint.file_exists": "File exists",
        "hint.file_not_exists": "File does not exist, please create",
        "hint.total_tools": "{count} tools total",
        "hint.set_as_default_role": "Set as default role",
        "hint.config_file": "Config file: {path}",
        "hint.no_available_mcp": "No available MCP servers",
        
        # 面板标题
        "panel.role_detail": "Role Details: {name}",
        "panel.recent_memory": "Recent Memory (Recent)",
        "panel.medium_memory": "Medium Memory (Medium)",
        "panel.long_memory": "Long Memory (Long)",
        
        # 角色详情
        "detail.name": "Name:",
        "detail.system_prompt": "System Prompt:",
        "detail.bind_model": "Bound Model:",
        "detail.is_default": "Default Role:",
        "detail.created_at": "Created At:",
        "detail.memory_status": "Memory Status:",
        "detail.recent_turns": "Recent memory: {count} turns",
        "detail.medium_count": "Medium memory: {count} summaries",
        "detail.long_status": "Long memory:",
        "detail.yes": "Yes",
        "detail.no": "No",
        "detail.none": "None",
        "detail.has": "Has",
        "detail.no_data": "None",
        "detail.unknown": "Unknown",
        
        # 记忆显示
        "memory.user": "User",
        "memory.assistant": "Assistant",
        "memory.turns_info": "({turns} turns, {time})",
    },
    
    "zh-cn": {
        # CLI 帮助
        "cli.description": "ask.py - 终端 LLM 问答工具",
        "cli.quick_ask": "快速提问: ask \"你的问题\"",
        "cli.use_role": "使用角色: ask -r coder \"帮我写代码\"",
        "cli.use_tools": "使用工具: ask -t \"帮我查询天气\"",
        "cli.manage_model": "管理模型: ask model --help",
        "cli.manage_role": "管理角色: ask role --help",
        "cli.manage_mcp": "管理工具: ask mcp --help",
        
        # 命令描述
        "cmd.q.desc": "向 LLM 提问",
        "cmd.q.example": """示例:
  ask "什么是机器学习?"
  ask -r coder "写一个快速排序"
  ask -t "帮我读取当前目录的文件"
  ask --mcp filesystem "列出 /tmp 目录\"""",
        "cmd.model.desc": "管理模型",
        "cmd.model.add.desc": "添加模型配置",
        "cmd.model.remove.desc": "删除模型配置",
        "cmd.model.default.desc": "设置默认模型",
        "cmd.model.list.desc": "列出所有模型配置",
        "cmd.role.desc": "管理角色",
        "cmd.role.add.desc": "添加新角色",
        "cmd.role.add.example": """示例:
  ask role add coder -s "你是一个资深程序员，擅长写简洁高效的代码"
  ask role add translator -s "你是专业翻译，精通中英文互译" --set-default""",
        "cmd.role.remove.desc": "删除角色",
        "cmd.role.list.desc": "列出所有角色",
        "cmd.role.show.desc": "显示角色详情",
        "cmd.role.default.desc": "设置或清除默认角色",
        "cmd.role.edit.desc": "编辑角色配置",
        "cmd.role.memory.desc": "查看角色记忆",
        "cmd.role.clear_memory.desc": "清除角色记忆",
        "cmd.mcp.desc": "查看 MCP 服务器状态",
        "cmd.mcp.config_hint": "MCP 配置文件: ~/.config/ask/mcp.json\n\n编辑配置文件添加/修改 MCP 服务器",
        "cmd.mcp.list.desc": "列出所有 MCP 服务器",
        "cmd.mcp.tools.desc": "列出 MCP 服务器提供的工具",
        "cmd.mcp.path.desc": "显示 MCP 配置文件路径",
        
        # 选项
        "opt.model": "指定使用的模型名称",
        "opt.system": "设置系统提示词（临时）",
        "opt.role": "使用指定角色",
        "opt.tools": "启用 MCP 工具（使用默认启用的服务器）",
        "opt.mcp": "指定 MCP 服务器（可多次使用）",
        "opt.api_base": "API 基础 URL",
        "opt.api_key": "API Key",
        "opt.model_name": "模型名称",
        "opt.temperature": "温度参数",
        "opt.set_default": "设置为默认",
        "opt.system_prompt": "系统提示词",
        "opt.bind_model": "绑定的模型（可选）",
        "opt.keep_memory": "保留记忆数据",
        "opt.confirm": "确认操作",
        "opt.level": "显示的记忆层级",
        "opt.new_system": "新的系统提示词",
        "opt.new_model": "绑定的模型（使用 'none' 清除）",
        
        # 表格标题
        "table.model_list": "模型配置列表",
        "table.col.name": "名称",
        "table.col.api_base": "API Base",
        "table.col.model": "模型",
        "table.col.default": "默认",
        "table.role_list": "角色列表",
        "table.col.system_prompt": "系统提示词",
        "table.col.bind_model": "绑定模型",
        "table.col.memory_turns": "记忆轮数",
        "table.mcp_list": "MCP 服务器列表",
        "table.col.command": "命令",
        "table.col.enabled": "启用",
        "table.mcp_tools": "MCP 工具 ({name})",
        "table.col.tool_name": "工具名称",
        "table.col.description": "描述",
        
        # 成功消息
        "success.model_added": "模型 '{name}' 已添加",
        "success.model_removed": "模型 '{name}' 已删除",
        "success.model_default": "默认模型已设置为 '{name}'",
        "success.set_as_default": "已设为默认",
        "success.role_created": "角色 '{name}' 已创建",
        "success.role_removed": "角色 '{name}' 已删除",
        "success.role_default_set": "默认角色已设置为 '{name}'",
        "success.role_default_cleared": "已清除默认角色",
        "success.memory_cleared": "角色 '{name}' 的记忆已清除",
        "success.memory_deleted": "记忆数据已删除",
        "success.system_updated": "系统提示词已更新",
        "success.model_bound": "已绑定模型 '{name}'",
        "success.model_unbound": "已清除绑定模型",
        
        # 错误消息
        "error.no_model_config": "尚未配置任何模型，请先运行 'ask model add' 添加模型",
        "error.model_not_found": "模型 '{name}' 不存在",
        "error.role_not_found": "角色 '{name}' 不存在",
        "error.mcp_not_installed": "MCP 支持未安装，请运行: uv sync",
        "error.mcp_server_not_found": "MCP 服务器 '{name}' 不存在",
        "error.mcp_connect_failed": "连接 MCP 服务器 '{name}' 失败: {error}",
        "error.no_mcp_servers": "没有可用的 MCP 服务器，请配置 ~/.config/ask/mcp.json",
        "error.no_mcp_tools": "MCP 服务器没有提供任何工具",
        "error.tool_not_found": "工具 '{name}' 在任何服务器中都未找到",
        "error.max_iterations": "达到最大迭代次数，请简化问题",
        "error.tool_error": "工具执行错误: {error}",
        "error.general": "错误: {error}",
        
        # 状态消息
        "status.thinking": "正在思考",
        "status.thinking_with_model": "正在思考 (模型: {model})",
        "status.thinking_with_role": "正在思考 (模型: {model}, 角色: {role})",
        "status.thinking_tools": "正在思考 (模型: {model}, 工具模式)",
        "status.connecting_mcp": "正在连接 MCP 服务器 '{name}'...",
        "status.compressing": "正在压缩记忆...",
        "status.compressed": "记忆压缩完成",
        "status.tool_call": "调用工具: {name}",
        "status.tool_success": "执行成功",
        "status.shell_command": "Shell 命令: {cmd}",
        "status.shell_confirm": "是否执行此命令? [y/N]: ",
        "status.shell_cancelled": "命令已取消",
        "status.shell_output": "输出:",
        
        # Shell 相关
        "error.shell_not_installed": "Shell 工具未安装，请运行: uv sync",
        "error.shell_rejected": "用户拒绝执行命令",
        
        # 提示消息
        "hint.no_model": "尚未配置任何模型",
        "hint.no_role": "尚未创建任何角色",
        "hint.create_role": "使用 'ask role add' 创建角色",
        "hint.no_mcp": "尚未配置任何 MCP 服务器",
        "hint.edit_config": "编辑配置文件: {path}",
        "hint.example_config": "示例配置:",
        "hint.confirm_clear": "将清除角色 '{name}' 的所有记忆数据",
        "hint.add_confirm": "添加 --confirm 参数确认操作",
        "hint.file_exists": "文件已存在",
        "hint.file_not_exists": "文件不存在，请创建",
        "hint.total_tools": "共 {count} 个工具",
        "hint.set_as_default_role": "已设为默认角色",
        "hint.config_file": "配置文件: {path}",
        "hint.no_available_mcp": "没有可用的 MCP 服务器",
        
        # 面板标题
        "panel.role_detail": "角色详情: {name}",
        "panel.recent_memory": "短期记忆 (Recent)",
        "panel.medium_memory": "中期记忆 (Medium)",
        "panel.long_memory": "长期记忆 (Long)",
        
        # 角色详情
        "detail.name": "名称:",
        "detail.system_prompt": "系统提示词:",
        "detail.bind_model": "绑定模型:",
        "detail.is_default": "默认角色:",
        "detail.created_at": "创建时间:",
        "detail.memory_status": "记忆状态:",
        "detail.recent_turns": "短期记忆: {count} 轮对话",
        "detail.medium_count": "中期记忆: {count} 条摘要",
        "detail.long_status": "长期记忆:",
        "detail.yes": "是",
        "detail.no": "否",
        "detail.none": "无",
        "detail.has": "有",
        "detail.no_data": "无",
        "detail.unknown": "未知",
        
        # 记忆显示
        "memory.user": "用户",
        "memory.assistant": "助手",
        "memory.turns_info": "({turns} 轮对话, {time})",
    },
    
    "zh-tw": {
        # CLI 帮助
        "cli.description": "ask.py - 終端 LLM 問答工具",
        "cli.quick_ask": "快速提問: ask \"你的問題\"",
        "cli.use_role": "使用角色: ask -r coder \"幫我寫程式\"",
        "cli.use_tools": "使用工具: ask -t \"幫我查詢天氣\"",
        "cli.manage_model": "管理模型: ask model --help",
        "cli.manage_role": "管理角色: ask role --help",
        "cli.manage_mcp": "管理工具: ask mcp --help",
        
        # 命令描述
        "cmd.q.desc": "向 LLM 提問",
        "cmd.q.example": """範例:
  ask "什麼是機器學習?"
  ask -r coder "寫一個快速排序"
  ask -t "幫我讀取當前目錄的檔案"
  ask --mcp filesystem "列出 /tmp 目錄\"""",
        "cmd.model.desc": "管理模型",
        "cmd.model.add.desc": "新增模型配置",
        "cmd.model.remove.desc": "刪除模型配置",
        "cmd.model.default.desc": "設定預設模型",
        "cmd.model.list.desc": "列出所有模型配置",
        "cmd.role.desc": "管理角色",
        "cmd.role.add.desc": "新增角色",
        "cmd.role.add.example": """範例:
  ask role add coder -s "你是一個資深程式設計師，擅長寫簡潔高效的程式碼"
  ask role add translator -s "你是專業翻譯，精通中英文互譯" --set-default""",
        "cmd.role.remove.desc": "刪除角色",
        "cmd.role.list.desc": "列出所有角色",
        "cmd.role.show.desc": "顯示角色詳情",
        "cmd.role.default.desc": "設定或清除預設角色",
        "cmd.role.edit.desc": "編輯角色配置",
        "cmd.role.memory.desc": "查看角色記憶",
        "cmd.role.clear_memory.desc": "清除角色記憶",
        "cmd.mcp.desc": "查看 MCP 伺服器狀態",
        "cmd.mcp.config_hint": "MCP 配置檔案: ~/.config/ask/mcp.json\n\n編輯配置檔案新增/修改 MCP 伺服器",
        "cmd.mcp.list.desc": "列出所有 MCP 伺服器",
        "cmd.mcp.tools.desc": "列出 MCP 伺服器提供的工具",
        "cmd.mcp.path.desc": "顯示 MCP 配置檔案路徑",
        
        # 選項
        "opt.model": "指定使用的模型名稱",
        "opt.system": "設定系統提示詞（臨時）",
        "opt.role": "使用指定角色",
        "opt.tools": "啟用 MCP 工具（使用預設啟用的伺服器）",
        "opt.mcp": "指定 MCP 伺服器（可多次使用）",
        "opt.api_base": "API 基礎 URL",
        "opt.api_key": "API Key",
        "opt.model_name": "模型名稱",
        "opt.temperature": "溫度參數",
        "opt.set_default": "設為預設",
        "opt.system_prompt": "系統提示詞",
        "opt.bind_model": "綁定的模型（可選）",
        "opt.keep_memory": "保留記憶資料",
        "opt.confirm": "確認操作",
        "opt.level": "顯示的記憶層級",
        "opt.new_system": "新的系統提示詞",
        "opt.new_model": "綁定的模型（使用 'none' 清除）",
        
        # 表格標題
        "table.model_list": "模型配置列表",
        "table.col.name": "名稱",
        "table.col.api_base": "API Base",
        "table.col.model": "模型",
        "table.col.default": "預設",
        "table.role_list": "角色列表",
        "table.col.system_prompt": "系統提示詞",
        "table.col.bind_model": "綁定模型",
        "table.col.memory_turns": "記憶輪數",
        "table.mcp_list": "MCP 伺服器列表",
        "table.col.command": "命令",
        "table.col.enabled": "啟用",
        "table.mcp_tools": "MCP 工具 ({name})",
        "table.col.tool_name": "工具名稱",
        "table.col.description": "描述",
        
        # 成功訊息
        "success.model_added": "模型 '{name}' 已新增",
        "success.model_removed": "模型 '{name}' 已刪除",
        "success.model_default": "預設模型已設定為 '{name}'",
        "success.set_as_default": "已設為預設",
        "success.role_created": "角色 '{name}' 已建立",
        "success.role_removed": "角色 '{name}' 已刪除",
        "success.role_default_set": "預設角色已設定為 '{name}'",
        "success.role_default_cleared": "已清除預設角色",
        "success.memory_cleared": "角色 '{name}' 的記憶已清除",
        "success.memory_deleted": "記憶資料已刪除",
        "success.system_updated": "系統提示詞已更新",
        "success.model_bound": "已綁定模型 '{name}'",
        "success.model_unbound": "已清除綁定模型",
        
        # 錯誤訊息
        "error.no_model_config": "尚未配置任何模型，請先執行 'ask model add' 新增模型",
        "error.model_not_found": "模型 '{name}' 不存在",
        "error.role_not_found": "角色 '{name}' 不存在",
        "error.mcp_not_installed": "MCP 支援未安裝，請執行: uv sync",
        "error.mcp_server_not_found": "MCP 伺服器 '{name}' 不存在",
        "error.mcp_connect_failed": "連接 MCP 伺服器 '{name}' 失敗: {error}",
        "error.no_mcp_servers": "沒有可用的 MCP 伺服器，請配置 ~/.config/ask/mcp.json",
        "error.no_mcp_tools": "MCP 伺服器沒有提供任何工具",
        "error.tool_not_found": "工具 '{name}' 在任何伺服器中都未找到",
        "error.max_iterations": "達到最大迭代次數，請簡化問題",
        "error.tool_error": "工具執行錯誤: {error}",
        "error.general": "錯誤: {error}",
        
        # 狀態訊息
        "status.thinking": "正在思考",
        "status.thinking_with_model": "正在思考 (模型: {model})",
        "status.thinking_with_role": "正在思考 (模型: {model}, 角色: {role})",
        "status.thinking_tools": "正在思考 (模型: {model}, 工具模式)",
        "status.connecting_mcp": "正在連接 MCP 伺服器 '{name}'...",
        "status.compressing": "正在壓縮記憶...",
        "status.compressed": "記憶壓縮完成",
        "status.tool_call": "呼叫工具: {name}",
        "status.tool_success": "執行成功",
        "status.shell_command": "Shell 命令: {cmd}",
        "status.shell_confirm": "是否執行此命令? [y/N]: ",
        "status.shell_cancelled": "命令已取消",
        "status.shell_output": "輸出:",
        
        # Shell 相關
        "error.shell_not_installed": "Shell 工具未安裝，請執行: uv sync",
        "error.shell_rejected": "使用者拒絕執行命令",
        
        # 提示訊息
        "hint.no_model": "尚未配置任何模型",
        "hint.no_role": "尚未建立任何角色",
        "hint.create_role": "使用 'ask role add' 建立角色",
        "hint.no_mcp": "尚未配置任何 MCP 伺服器",
        "hint.edit_config": "編輯配置檔案: {path}",
        "hint.example_config": "範例配置:",
        "hint.confirm_clear": "將清除角色 '{name}' 的所有記憶資料",
        "hint.add_confirm": "新增 --confirm 參數確認操作",
        "hint.file_exists": "檔案已存在",
        "hint.file_not_exists": "檔案不存在，請建立",
        "hint.total_tools": "共 {count} 個工具",
        "hint.set_as_default_role": "已設為預設角色",
        "hint.config_file": "配置檔案: {path}",
        "hint.no_available_mcp": "沒有可用的 MCP 伺服器",
        
        # 面板標題
        "panel.role_detail": "角色詳情: {name}",
        "panel.recent_memory": "短期記憶 (Recent)",
        "panel.medium_memory": "中期記憶 (Medium)",
        "panel.long_memory": "長期記憶 (Long)",
        
        # 角色詳情
        "detail.name": "名稱:",
        "detail.system_prompt": "系統提示詞:",
        "detail.bind_model": "綁定模型:",
        "detail.is_default": "預設角色:",
        "detail.created_at": "建立時間:",
        "detail.memory_status": "記憶狀態:",
        "detail.recent_turns": "短期記憶: {count} 輪對話",
        "detail.medium_count": "中期記憶: {count} 條摘要",
        "detail.long_status": "長期記憶:",
        "detail.yes": "是",
        "detail.no": "否",
        "detail.none": "無",
        "detail.has": "有",
        "detail.no_data": "無",
        "detail.unknown": "未知",
        
        # 記憶顯示
        "memory.user": "使用者",
        "memory.assistant": "助手",
        "memory.turns_info": "({turns} 輪對話, {time})",
    },
    
    "ja": {
        # CLI ヘルプ
        "cli.description": "ask.py - ターミナル LLM Q&A ツール",
        "cli.quick_ask": "質問: ask \"あなたの質問\"",
        "cli.use_role": "ロール使用: ask -r coder \"コードを書いて\"",
        "cli.use_tools": "ツール使用: ask -t \"天気を調べて\"",
        "cli.manage_model": "モデル管理: ask model --help",
        "cli.manage_role": "ロール管理: ask role --help",
        "cli.manage_mcp": "ツール管理: ask mcp --help",
        
        # コマンド説明
        "cmd.q.desc": "LLM に質問する",
        "cmd.q.example": """例:
  ask "機械学習とは?"
  ask -r coder "クイックソートを書いて"
  ask -t "現在のディレクトリのファイルを読んで"
  ask --mcp filesystem "/tmp ディレクトリを一覧表示\"""",
        "cmd.model.desc": "モデル管理",
        "cmd.model.add.desc": "モデル設定を追加",
        "cmd.model.remove.desc": "モデル設定を削除",
        "cmd.model.default.desc": "デフォルトモデルを設定",
        "cmd.model.list.desc": "すべてのモデル設定を一覧表示",
        "cmd.role.desc": "ロール管理",
        "cmd.role.add.desc": "新しいロールを追加",
        "cmd.role.add.example": """例:
  ask role add coder -s "あなたは経験豊富なプログラマーです"
  ask role add translator -s "あなたはプロの翻訳者です" --set-default""",
        "cmd.role.remove.desc": "ロールを削除",
        "cmd.role.list.desc": "すべてのロールを一覧表示",
        "cmd.role.show.desc": "ロールの詳細を表示",
        "cmd.role.default.desc": "デフォルトロールを設定/解除",
        "cmd.role.edit.desc": "ロール設定を編集",
        "cmd.role.memory.desc": "ロールの記憶を表示",
        "cmd.role.clear_memory.desc": "ロールの記憶をクリア",
        "cmd.mcp.desc": "MCP サーバーの状態を表示",
        "cmd.mcp.config_hint": "MCP 設定ファイル: ~/.config/ask/mcp.json\n\n設定ファイルを編集して MCP サーバーを追加/変更",
        "cmd.mcp.list.desc": "すべての MCP サーバーを一覧表示",
        "cmd.mcp.tools.desc": "MCP サーバーが提供するツールを一覧表示",
        "cmd.mcp.path.desc": "MCP 設定ファイルのパスを表示",
        
        # オプション
        "opt.model": "使用するモデル名を指定",
        "opt.system": "システムプロンプトを設定（一時的）",
        "opt.role": "指定したロールを使用",
        "opt.tools": "MCP ツールを有効化（デフォルト有効のサーバーを使用）",
        "opt.mcp": "MCP サーバーを指定（複数回使用可）",
        "opt.api_base": "API ベース URL",
        "opt.api_key": "API キー",
        "opt.model_name": "モデル名",
        "opt.temperature": "温度パラメータ",
        "opt.set_default": "デフォルトに設定",
        "opt.system_prompt": "システムプロンプト",
        "opt.bind_model": "バインドするモデル（オプション）",
        "opt.keep_memory": "記憶データを保持",
        "opt.confirm": "操作を確認",
        "opt.level": "表示する記憶レベル",
        "opt.new_system": "新しいシステムプロンプト",
        "opt.new_model": "バインドするモデル（'none' でクリア）",
        
        # テーブルタイトル
        "table.model_list": "モデル設定一覧",
        "table.col.name": "名前",
        "table.col.api_base": "API Base",
        "table.col.model": "モデル",
        "table.col.default": "デフォルト",
        "table.role_list": "ロール一覧",
        "table.col.system_prompt": "システムプロンプト",
        "table.col.bind_model": "バインドモデル",
        "table.col.memory_turns": "記憶ターン数",
        "table.mcp_list": "MCP サーバー一覧",
        "table.col.command": "コマンド",
        "table.col.enabled": "有効",
        "table.mcp_tools": "MCP ツール ({name})",
        "table.col.tool_name": "ツール名",
        "table.col.description": "説明",
        
        # 成功メッセージ
        "success.model_added": "モデル '{name}' を追加しました",
        "success.model_removed": "モデル '{name}' を削除しました",
        "success.model_default": "デフォルトモデルを '{name}' に設定しました",
        "success.set_as_default": "デフォルトに設定しました",
        "success.role_created": "ロール '{name}' を作成しました",
        "success.role_removed": "ロール '{name}' を削除しました",
        "success.role_default_set": "デフォルトロールを '{name}' に設定しました",
        "success.role_default_cleared": "デフォルトロールをクリアしました",
        "success.memory_cleared": "ロール '{name}' の記憶をクリアしました",
        "success.memory_deleted": "記憶データを削除しました",
        "success.system_updated": "システムプロンプトを更新しました",
        "success.model_bound": "モデル '{name}' をバインドしました",
        "success.model_unbound": "モデルバインドをクリアしました",
        
        # エラーメッセージ
        "error.no_model_config": "モデルが設定されていません。先に 'ask model add' を実行してください",
        "error.model_not_found": "モデル '{name}' が存在しません",
        "error.role_not_found": "ロール '{name}' が存在しません",
        "error.mcp_not_installed": "MCP サポートがインストールされていません。uv sync を実行してください",
        "error.mcp_server_not_found": "MCP サーバー '{name}' が存在しません",
        "error.mcp_connect_failed": "MCP サーバー '{name}' への接続に失敗しました: {error}",
        "error.no_mcp_servers": "利用可能な MCP サーバーがありません。~/.config/ask/mcp.json を設定してください",
        "error.no_mcp_tools": "MCP サーバーがツールを提供していません",
        "error.tool_not_found": "ツール '{name}' はどのサーバーにも見つかりません",
        "error.max_iterations": "最大反復回数に達しました。質問を簡略化してください",
        "error.tool_error": "ツール実行エラー: {error}",
        "error.general": "エラー: {error}",
        
        # ステータスメッセージ
        "status.thinking": "考え中",
        "status.thinking_with_model": "考え中 (モデル: {model})",
        "status.thinking_with_role": "考え中 (モデル: {model}, ロール: {role})",
        "status.thinking_tools": "考え中 (モデル: {model}, ツールモード)",
        "status.connecting_mcp": "MCP サーバー '{name}' に接続中...",
        "status.compressing": "記憶を圧縮中...",
        "status.compressed": "記憶の圧縮が完了しました",
        "status.tool_call": "ツールを呼び出し中: {name}",
        "status.tool_success": "実行成功",
        "status.shell_command": "Shell コマンド: {cmd}",
        "status.shell_confirm": "このコマンドを実行しますか? [y/N]: ",
        "status.shell_cancelled": "コマンドがキャンセルされました",
        "status.shell_output": "出力:",
        
        # Shell 関連
        "error.shell_not_installed": "Shell ツールがインストールされていません。uv sync を実行してください",
        "error.shell_rejected": "ユーザーがコマンドを拒否しました",
        
        # ヒントメッセージ
        "hint.no_model": "モデルが設定されていません",
        "hint.no_role": "ロールが作成されていません",
        "hint.create_role": "'ask role add' でロールを作成してください",
        "hint.no_mcp": "MCP サーバーが設定されていません",
        "hint.edit_config": "設定ファイルを編集: {path}",
        "hint.example_config": "設定例:",
        "hint.confirm_clear": "ロール '{name}' のすべての記憶データをクリアします",
        "hint.add_confirm": "--confirm を追加して操作を確認してください",
        "hint.file_exists": "ファイルが存在します",
        "hint.file_not_exists": "ファイルが存在しません。作成してください",
        "hint.total_tools": "合計 {count} 個のツール",
        "hint.set_as_default_role": "デフォルトロールに設定しました",
        "hint.config_file": "設定ファイル: {path}",
        "hint.no_available_mcp": "利用可能な MCP サーバーがありません",
        
        # パネルタイトル
        "panel.role_detail": "ロール詳細: {name}",
        "panel.recent_memory": "短期記憶 (Recent)",
        "panel.medium_memory": "中期記憶 (Medium)",
        "panel.long_memory": "長期記憶 (Long)",
        
        # ロール詳細
        "detail.name": "名前:",
        "detail.system_prompt": "システムプロンプト:",
        "detail.bind_model": "バインドモデル:",
        "detail.is_default": "デフォルトロール:",
        "detail.created_at": "作成日時:",
        "detail.memory_status": "記憶状態:",
        "detail.recent_turns": "短期記憶: {count} ターン",
        "detail.medium_count": "中期記憶: {count} 件の要約",
        "detail.long_status": "長期記憶:",
        "detail.yes": "はい",
        "detail.no": "いいえ",
        "detail.none": "なし",
        "detail.has": "あり",
        "detail.no_data": "なし",
        "detail.unknown": "不明",
        
        # 記憶表示
        "memory.user": "ユーザー",
        "memory.assistant": "アシスタント",
        "memory.turns_info": "({turns} ターン, {time})",
    },
}


# ==================== 语言检测与管理 ====================

def detect_lang_from_env() -> str:
    """从环境变量检测语言
    
    检测顺序: LANGUAGE -> LC_ALL -> LC_MESSAGES -> LANG
    """
    for var in ["LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"]:
        value = os.environ.get(var, "").lower()
        if not value:
            continue
        
        # 解析语言代码 (如 zh_CN.UTF-8 -> zh_cn)
        lang_code = value.split(".")[0].replace("-", "_")
        
        # 映射到支持的语言
        if lang_code.startswith("zh_cn") or lang_code == "zh":
            return "zh-cn"
        elif lang_code.startswith("zh_tw") or lang_code.startswith("zh_hk"):
            return "zh-tw"
        elif lang_code.startswith("ja"):
            return "ja"
        elif lang_code.startswith("en"):
            return "en"
    
    return DEFAULT_LANG


def get_lang() -> str:
    """获取当前语言"""
    global _current_lang
    if _current_lang is None:
        _current_lang = detect_lang_from_env()
    return _current_lang


def set_lang(lang: str) -> None:
    """设置当前语言
    
    Args:
        lang: 语言代码 (en, zh-cn, zh-tw)
    """
    global _current_lang
    if lang in SUPPORTED_LANGS:
        _current_lang = lang
    else:
        _current_lang = DEFAULT_LANG


def t(key: str, **kwargs) -> str:
    """获取翻译文本
    
    Args:
        key: 翻译键 (如 "error.model_not_found")
        **kwargs: 格式化参数
    
    Returns:
        翻译后的文本
    
    Example:
        t("error.model_not_found", name="gpt-4")
        # -> "模型 'gpt-4' 不存在"
    """
    lang = get_lang()
    translations = TRANSLATIONS.get(lang, TRANSLATIONS[DEFAULT_LANG])
    
    # 尝试获取翻译，如果不存在则回退到英文
    text = translations.get(key)
    if text is None:
        text = TRANSLATIONS[DEFAULT_LANG].get(key, key)
    
    # 格式化参数
    if kwargs:
        try:
            text = text.format(**kwargs)
        except KeyError:
            pass
    
    return text
