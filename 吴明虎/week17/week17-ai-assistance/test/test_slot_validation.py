#!/usr/bin/env python3
"""测试槽位值验证功能"""

import sys
sys.path.insert(0, '.')

from task_dialogue_system import DialogueSystem, NLU, SlotOntology
from pathlib import Path

def test_slot_validation():
    """测试槽位值验证"""
    print("=" * 50)
    print("测试槽位值验证功能")
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
    
    # 测试2: 输入有效的服装类型
    print("\n测试2: 输入有效的服装类型 '衬衫'")
    reply = ds.chat("衬衫")
    print(f"系统回复: {reply}")
    
    # 测试3: 输入无效的服装类型
    print("\n测试3: 输入无效的服装类型 '紫色'")
    reply = ds.chat("紫色")
    print(f"系统回复: {reply}")
    
    # 测试4: 输入有效的颜色
    print("\n测试4: 输入有效的颜色 '蓝色'")
    reply = ds.chat("蓝色")
    print(f"系统回复: {reply}")
    
    # 测试5: 输入有效的尺寸
    print("\n测试5: 输入有效的尺寸 'M'")
    reply = ds.chat("M")
    print(f"系统回复: {reply}")
    
    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)

if __name__ == "__main__":
    test_slot_validation()
