#!/usr/bin/env python
# coding: utf-8

"""
运行脚本 - 用于快速运行训练和测试
"""

import sys
import os

def main():
    """主函数"""
    print("=" * 60)
    print("三元组损失文本匹配模型")
    print("=" * 60)
    print()
    print("使用说明:")
    print("python run.py train    - 训练模型")
    print("python run.py test     - 测试模型")
    print("python run.py predict  - 预测模式")
    print()
    
    if len(sys.argv) < 2:
        print("请指定运行模式!")
        return
    
    mode = sys.argv[1]
    
    if mode == "train":
        print("开始训练模型...")
        from main import main as train_main
        train_main()

    elif mode == "predict":
        print("启动预测模式...")
        from predict import main as predict_main
        predict_main()
    
    else:
        print(f"未知模式: {mode}")
        print("支持的模式: train, test, predict")

if __name__ == "__main__":
    main()