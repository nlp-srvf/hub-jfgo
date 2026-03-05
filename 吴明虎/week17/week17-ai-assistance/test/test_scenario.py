from pathlib import Path
import json

# 测试场景文件加载
base_dir = Path(__file__).parent
scenario_dir = base_dir / "scenario"

print(f"场景目录: {scenario_dir}")
print(f"目录是否存在: {scenario_dir.exists()}")

# 列出目录中的文件
files = list(scenario_dir.glob("*.json"))
print(f"找到的JSON文件: {files}")

# 尝试加载一个场景文件
if files:
    test_file = files[0]
    print(f"\n测试加载文件: {test_file}")
    try:
        data = json.loads(test_file.read_text(encoding="utf-8"))
        print(f"加载成功，节点数量: {len(data)}")
        print(f"第一个节点: {data[0]}")
    except Exception as e:
        print(f"加载失败: {e}")
else:
    print("没有找到JSON文件")
