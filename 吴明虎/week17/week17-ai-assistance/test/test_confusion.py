#!/usr/bin/env python3
"""测试困惑语句检测功能"""

import sys
sys.path.insert(0, '.')

from task_dialogue_system import DialogueSystem
from pathlib import Path

def test_confusion():
    """测试困惑语句检测"""
    print("=" * 50)
    print("测试困惑语句检测功能")
    print("=" * 50)
    
    # 创建对话系统
    base_dir = Path(".")
    ds = DialogueSystem(base_dir / "scenario")
    
    print(f"\n可用场景: {ds.list_scenarios()}")
    
    # 测试买衣服场景
    print("\n" + "=" * 50)
    print("测试场景: 买衣服")
    print("=" * 50)
    
    ds.start("买衣服")
    
    # 测试1: 正常输入
    print("\n测试1: 正常输入 '我要买衣服'")
    reply = ds.chat("我要买衣服")
    print(f"系统回复: {reply}")
    
    # 测试2: 输入困惑语句
    print("\n测试2: 输入困惑语句 '什么？'")
    reply = ds.chat("什么？")
    print(f"系统回复: {reply}")
    
    # 测试3: 输入另一个困惑语句
    print("\n测试3: 输入困惑语句 '怎么了？'")
    reply = ds.chat("怎么了？")
    print(f"系统回复: {reply}")
    
    # 测试4: 输入困惑语句
    print("\n测试4: 输入困惑语句 '不明白'")
    reply = ds.chat("不明白")
    print(f"系统回复: {reply}")
    
    # 测试5: 正常回答后继续
    print("\n测试5: 正常输入 '衬衫'")
    reply = ds.chat("衬衫")
    print(f"系统回复: {reply}")
    
    # 测试6: 再次输入困惑语句
    print("\n测试6: 输入困惑语句 '什么意思'")
    reply = ds.chat("什么意思")
    print(f"系统回复: {reply}")
    
    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)

if __name__ == "__main__":
    test_confusion()
