#!/usr/bin/env python3
"""测试节点导航功能 - 验证node1-2-3可以相互跳转"""

import sys
sys.path.insert(0, '..')

from task_dialogue_system import DialogueSystem
from pathlib import Path

def test_node_navigation():
    """测试节点导航"""
    print("=" * 60)
    print("测试节点导航功能 - node1-2-3相互跳转")
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
    
    # 测试1: 正常输入开始
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
    
    # 测试4: 从node1跳到node3（可乐）
    print("\n测试4: 从node1跳到node3 '来个可乐'")
    reply = ds.chat("来个可乐")
    print(f"系统回复: {reply}")
    
    # 测试5: 从node3跳到node2（爆米花）- 验证相互跳转
    print("\n测试5: 从node3跳到node2 '来个爆米花'")
    reply = ds.chat("来个爆米花")
    print(f"系统回复: {reply}")
    
    # 测试6: 从node2跳到node1（重新选电影）- 验证相互跳转
    print("\n测试6: 从node2跳到node1 '我要看电影'")
    reply = ds.chat("我要看电影")
    print(f"系统回复: {reply}")
    
    # 测试7: 输入时间
    print("\n测试7: 输入时间 '9点'")
    reply = ds.chat("9点")
    print(f"系统回复: {reply}")
    
    # 测试8: 输入电影名称
    print("\n测试8: 输入电影名称 '泰坦尼克号'")
    reply = ds.chat("泰坦尼克号")
    print(f"系统回复: {reply}")
    
    # 测试9: 在node1输入非预期回复，检查是否提示所有功能
    print("\n测试9: 在node1输入非预期回复 '随便聊聊'")
    reply = ds.chat("随便聊聊")
    print(f"系统回复: {reply}")
    
    # 测试10: 在node3输入非预期回复，检查是否提示所有功能
    print("\n测试10: 跳到node3后输入非预期回复")
    ds.chat("来个可乐")  # 跳到node3
    print("  用户: 来个可乐")
    reply = ds.chat("今天天气不错")  # 非预期回复
    print(f"系统回复: {reply}")
    
    # 测试11: 正常结束
    print("\n测试11: 正常输入 '结束了'")
    reply = ds.chat("结束了")
    print(f"系统回复: {reply}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_node_navigation()
