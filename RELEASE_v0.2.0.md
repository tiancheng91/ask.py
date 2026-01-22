# Release v0.2.0

## 🚀 新功能

### 流式输出
- 默认启用流式输出，实时显示回答内容
- 使用 `rich.live` 实现实时 Markdown 渲染
- 支持 `--no-stream` 禁用流式输出

### 上下文感知
- 自动注入当前工作目录
- 自动注入系统信息（操作系统、Python 版本）
- 自动注入重要环境变量（PATH, HOME, USER, SHELL, LANG, EDITOR）
- 无需手动配置，系统自动识别并添加到上下文中

### 文件内容分析
- 支持 `-f/--file` 参数读取文件内容
- 支持分析代码文件、配置文件、日志文件等
- 文件内容自动添加到问题上下文中

### 错误日志分析
- 支持 `--stdin` 从标准输入读取内容
- 支持管道输入：`cat error.log | ask "分析这个错误" --stdin`
- 适合分析错误日志、命令输出等

## ⚡ 性能优化

- 实现 MCP 连接复用：单次执行内复用连接，避免重复建立连接
- 并行连接多个 MCP 服务器：使用 `asyncio.gather` 并行连接
- 延迟导入 MCP 模块：只在需要时导入，不影响 `--help` 等命令
- 添加配置加载缓存：使用 `@lru_cache` 缓存 `load_config/load_roles/load_mcp_config`
- 缓存包运行器检测结果：避免重复检测 uvx/pipx

## 🐛 修复

- 移除 shell 服务器的默认启用状态（由于执行准确性问题）
- 修复重复的 `@lru_cache` 装饰器

## 📝 文档更新

- 更新 README 示例，添加新功能使用说明
- 更新角色示例为 `shell` 角色，强调优先使用 shell 命令解决系统问题
- 添加详细的使用示例和场景说明

## 📦 使用示例

```bash
# 流式输出（默认）
ask "什么是 Python？"

# 分析文件
ask -f main.py "解释这个文件"

# 分析错误日志
cat error.log | ask "分析这个错误" --stdin

# 使用 shell 角色
ask role add shell -s "你是一个系统管理员助手。当用户询问系统相关问题（如文件操作、进程管理、系统信息查询等）时，优先使用 shell 命令解决，而不是使用其他编程语言代码实现。" --set-default
ask "找出占用 CPU 最高的进程"
```

## 🔗 相关链接

- [完整更新日志](https://github.com/tiancheng91/ask.py/compare/v0.1.2...v0.2.0)
- [文档](https://github.com/tiancheng91/ask.py#readme)
