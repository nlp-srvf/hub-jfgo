#!/usr/bin/env python3
"""测试非预期回复处理功能"""

import sys
sys.path.insert(0, '..')

from task_dialogue_system import DialogueSystem
from pathlib import Path

def test_unexpected_reply():
    """测试非预期回复处理"""
    print("=" * 60)
    print("测试非预期回复处理功能")
    print("=" * 60)
    
    # 创建对话系统
    base_dir = Path("..")
    ds = DialogueSystem(base_dir / "scenario")
    
    print(f"\n可用场景: {ds.list_scenarios()}")
    
    # 测试看电影场景
    print("\n" + "=" * 60)
    print("测试场景: 看电影")
    print("=" * 60)
    
    ds.start("看电影")
    
    # 测试1: 正常输入
    print("\n测试1: 正常输入 '我要看电影'")
    reply = ds.chat("我要看电影")
    print(f"系统回复: {reply}")
    
    # 测试2: 输入时间
    print("\n测试2: 输入时间 '8点'")
    reply = ds.chat("8点")
    print(f"系统回复: {reply}")
    
    # 测试3: 输入电影名称
    print("\n测试3: 输入电影名称 '阿凡达'")
    reply = ds.chat("阿凡达")
    print(f"系统回复: {reply}")
    
    # 测试4: 输入非预期回复（不在任何意图范围内，且不包含困惑关键词）
    print("\n测试4: 输入非预期回复 '随便聊聊'")
    reply = ds.chat("随便聊聊")
    print(f"系统回复: {reply}")
    
    # 测试5: 输入另一个非预期回复
    print("\n测试5: 输入非预期回复 '你好'")
    reply = ds.chat("你好")
    print(f"系统回复: {reply}")
    
    # 测试6: 正常输入爆米花
    print("\n测试6: 正常输入 '来个爆米花'")
    reply = ds.chat("来个爆米花")
    print(f"系统回复: {reply}")
    
    # 测试7: 再次输入非预期回复
    print("\n测试7: 输入非预期回复 '今天天气不错'")
    reply = ds.chat("今天天气不错")
    print(f"系统回复: {reply}")
    
    # 测试8: 正常结束
    print("\n测试8: 正常输入 '结束了'")
    reply = ds.chat("结束了")
    print(f"系统回复: {reply}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_unexpected_reply()
