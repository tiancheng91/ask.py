#!/usr/bin/env python3
"""
ask.py 测试文件

环境变量配置（兼容 OpenAI SDK）：
  OPENAI_API_KEY    - API 密钥
  OPENAI_API_BASE   - API 基础 URL (默认: https://api.openai.com/v1)
  OPENAI_MODEL      - 模型名称 (默认: gpt-3.5-turbo)

运行测试：
  uv run pytest test_ask.py -v
  
  # 或设置环境变量
  OPENAI_API_KEY=sk-xxx uv run pytest test_ask.py -v
"""

# 关闭 urllib3 SSL 警告
import warnings
warnings.filterwarnings("ignore", message=".*urllib3.*OpenSSL.*")
warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")

import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest import TestCase, main, skipIf
import pytest

# 设置测试环境的配置目录
TEST_CONFIG_DIR = Path(tempfile.mkdtemp(prefix="ask_test_"))
os.environ["ASK_CONFIG_DIR"] = str(TEST_CONFIG_DIR)

# 在导入 ask 模块前修改配置目录
import ask
ask.CONFIG_DIR = TEST_CONFIG_DIR
ask.CONFIG_FILE = TEST_CONFIG_DIR / "config.yaml"
ask.ROLES_FILE = TEST_CONFIG_DIR / "roles.yaml"
ask.MEMORY_DIR = TEST_CONFIG_DIR / "memory"
ask.MCP_FILE = TEST_CONFIG_DIR / "mcp.json"


def get_test_config():
    """从环境变量获取测试配置"""
    return {
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "api_base": os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
        "model": os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"),
    }


def has_api_config():
    """检查是否配置了 API"""
    return bool(os.environ.get("OPENAI_API_KEY"))


class TestConfigManagement(TestCase):
    """测试配置管理功能"""
    
    def setUp(self):
        """每个测试前清理配置"""
        if ask.CONFIG_FILE.exists():
            ask.CONFIG_FILE.unlink()
        if ask.ROLES_FILE.exists():
            ask.ROLES_FILE.unlink()
    
    def test_load_empty_config(self):
        """测试加载空配置"""
        config = ask.load_config()
        self.assertEqual(config["models"], {})
        self.assertIsNone(config["default"])
    
    def test_save_and_load_config(self):
        """测试保存和加载配置"""
        config = {
            "models": {
                "test": {
                    "api_base": "https://api.test.com/v1",
                    "api_key": "sk-test",
                    "model": "test-model",
                    "temperature": 0.5,
                }
            },
            "default": "test",
            "default_role": None,
        }
        ask.save_config(config)
        
        loaded = ask.load_config()
        self.assertEqual(loaded["models"]["test"]["api_base"], "https://api.test.com/v1")
        self.assertEqual(loaded["default"], "test")
    
    def test_add_multiple_models(self):
        """测试添加多个模型"""
        config = ask.load_config()
        
        config["models"]["model1"] = {"api_base": "url1", "api_key": "key1", "model": "m1"}
        config["models"]["model2"] = {"api_base": "url2", "api_key": "key2", "model": "m2"}
        config["default"] = "model1"
        ask.save_config(config)
        
        loaded = ask.load_config()
        self.assertEqual(len(loaded["models"]), 2)
        self.assertIn("model1", loaded["models"])
        self.assertIn("model2", loaded["models"])


class TestRoleManagement(TestCase):
    """测试角色管理功能"""
    
    def setUp(self):
        """每个测试前清理配置"""
        if ask.ROLES_FILE.exists():
            ask.ROLES_FILE.unlink()
        # 清理记忆目录
        if ask.MEMORY_DIR.exists():
            shutil.rmtree(ask.MEMORY_DIR)
    
    def test_load_empty_roles(self):
        """测试加载空角色配置"""
        roles = ask.load_roles()
        self.assertEqual(roles, {})
    
    def test_save_and_load_roles(self):
        """测试保存和加载角色"""
        roles = {
            "coder": {
                "system_prompt": "你是程序员",
                "model": None,
                "created_at": "2024-01-01",
            }
        }
        ask.save_roles(roles)
        
        loaded = ask.load_roles()
        self.assertIn("coder", loaded)
        self.assertEqual(loaded["coder"]["system_prompt"], "你是程序员")


class TestMemoryManagement(TestCase):
    """测试记忆管理功能"""
    
    def setUp(self):
        """每个测试前清理记忆"""
        if ask.MEMORY_DIR.exists():
            shutil.rmtree(ask.MEMORY_DIR)
    
    def test_load_empty_memory(self):
        """测试加载空记忆"""
        memory = ask.load_memory("test_role")
        self.assertEqual(memory["recent"], [])
        self.assertEqual(memory["medium"], [])
        self.assertEqual(memory["long"], "")
    
    def test_add_to_memory(self):
        """测试添加记忆"""
        ask.add_to_memory("test_role", "你好", "你好！有什么可以帮助你的？")
        
        memory = ask.load_memory("test_role")
        self.assertEqual(len(memory["recent"]), 2)
        self.assertEqual(memory["recent"][0]["role"], "user")
        self.assertEqual(memory["recent"][0]["content"], "你好")
        self.assertEqual(memory["recent"][1]["role"], "assistant")
    
    def test_memory_accumulation(self):
        """测试记忆累积"""
        for i in range(5):
            ask.add_to_memory("test_role", f"问题{i}", f"回答{i}")
        
        memory = ask.load_memory("test_role")
        self.assertEqual(len(memory["recent"]), 10)  # 5轮 * 2
    
    def test_build_context_messages(self):
        """测试构建上下文消息"""
        ask.add_to_memory("test_role", "你好", "你好！")
        
        messages = ask.build_context_messages("test_role", "你是助手")
        
        # 应该包含: system + user + assistant + (新的 user 在外部添加)
        self.assertEqual(len(messages), 3)
        self.assertIn("你是助手", messages[0].content)


class TestMemoryCompression(TestCase):
    """测试记忆压缩功能"""
    
    def setUp(self):
        if ask.MEMORY_DIR.exists():
            shutil.rmtree(ask.MEMORY_DIR)
    
    def test_compression_threshold(self):
        """测试压缩阈值"""
        # 添加少于阈值的对话，不应触发压缩
        for i in range(5):
            ask.add_to_memory("test_role", f"问题{i}", f"回答{i}")
        
        memory = ask.load_memory("test_role")
        self.assertEqual(len(memory["recent"]), 10)
        self.assertEqual(len(memory["medium"]), 0)  # 未触发压缩


@skipIf(not has_api_config(), "跳过: 未设置 OPENAI_API_KEY 环境变量")
class TestLLMIntegration(TestCase):
    """测试 LLM 集成功能（需要 API 配置）"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试模型配置"""
        cfg = get_test_config()
        config = {
            "models": {
                "test": {
                    "api_base": cfg["api_base"],
                    "api_key": cfg["api_key"],
                    "model": cfg["model"],
                    "temperature": 0.7,
                }
            },
            "default": "test",
            "default_role": None,
        }
        ask.save_config(config)
    
    def test_get_model(self):
        """测试获取模型实例"""
        model = ask.get_model("test")
        self.assertIsNotNone(model)
    
    def test_simple_query(self):
        """测试简单问答"""
        from langchain_core.messages import HumanMessage
        
        model = ask.get_model("test")
        response = model.invoke([HumanMessage(content="回复OK")])
        
        self.assertIsNotNone(response)
        self.assertIsNotNone(response.content)
        print(f"\n模型回复: {response.content}")
    
    def test_query_with_role(self):
        """测试使用角色问答"""
        # 创建角色
        roles = {"test_coder": {"system_prompt": "简短回答，只说OK", "model": None}}
        ask.save_roles(roles)
        
        # 构建消息
        messages = ask.build_context_messages("test_coder", "简短回答，只说OK")
        messages.append(ask.HumanMessage(content="测试"))
        
        model = ask.get_model("test")
        response = model.invoke(messages)
        
        self.assertIsNotNone(response.content)
        print(f"\n角色回复: {response.content}")


@skipIf(not has_api_config(), "跳过: 未设置 OPENAI_API_KEY 环境变量")
class TestMemoryCompressionWithLLM(TestCase):
    """测试记忆压缩功能（需要 API）"""
    
    @classmethod
    def setUpClass(cls):
        cfg = get_test_config()
        config = {
            "models": {
                "test": {
                    "api_base": cfg["api_base"],
                    "api_key": cfg["api_key"],
                    "model": cfg["model"],
                    "temperature": 0.7,
                }
            },
            "default": "test",
            "default_role": None,
        }
        ask.save_config(config)
    
    def setUp(self):
        if ask.MEMORY_DIR.exists():
            shutil.rmtree(ask.MEMORY_DIR)
    
    def test_memory_compression(self):
        """测试记忆压缩"""
        # 添加超过阈值的对话
        for i in range(12):
            ask.add_to_memory("compress_test", f"这是第{i}个问题", f"这是第{i}个回答")
        
        # 触发压缩
        model = ask.get_model("test")
        ask.compress_memory("compress_test", model)
        
        memory = ask.load_memory("compress_test")
        
        # 验证压缩结果
        print(f"\n压缩后短期记忆: {len(memory['recent'])} 条")
        print(f"中期记忆: {len(memory['medium'])} 条")
        if memory["medium"]:
            print(f"摘要内容: {memory['medium'][0]['summary']}")
        
        self.assertLessEqual(len(memory["recent"]), ask.MEMORY_CONFIG["recent_limit"] * 2)


def cleanup():
    """清理测试目录"""
    if TEST_CONFIG_DIR.exists():
        shutil.rmtree(TEST_CONFIG_DIR)


if __name__ == "__main__":
    try:
        print("=" * 60)
        print("ask.py 测试")
        print("=" * 60)
        
        cfg = get_test_config()
        if has_api_config():
            print(f"API Base: {cfg['api_base']}")
            print(f"Model: {cfg['model']}")
            print(f"API Key: {cfg['api_key'][:8]}...")
        else:
            print("⚠️  未设置 OPENAI_API_KEY，将跳过 LLM 集成测试")
        
        print(f"测试配置目录: {TEST_CONFIG_DIR}")
        print("=" * 60)
        
        main(verbosity=2, exit=False)
    finally:
        cleanup()
        print(f"\n已清理测试目录: {TEST_CONFIG_DIR}")
