import random


def split_file(input_file, output_file1, output_file2, split_ratio=0.8, seed=None):
    """
    随机分割文件为两部分

    参数:
    input_file: 输入文件路径
    output_file1: 输出文件1路径（包含split_ratio比例的行）
    output_file2: 输出文件2路径（包含剩余比例的行）
    split_ratio: 分割比例，默认为0.8
    seed: 随机种子，用于可重复的结果
    """

    # 设置随机种子（如果需要可重复的结果）
    if seed is not None:
        random.seed(seed)

    try:
        # 读取所有行
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 创建行索引列表
        lines = lines[1:]
        indices = list(range(len(lines)))

        # 随机打乱索引
        random.shuffle(indices)

        # 计算分割点
        split_point = int(len(lines) * split_ratio)

        # 分割索引
        indices1 = indices[:split_point]  # 80%的部分
        indices2 = indices[split_point:]  # 20%的部分

        # 按照原始顺序写入文件（可选，保持原顺序）
        # 如果需要保持随机顺序，可以跳过排序
        indices1.sort()
        indices2.sort()

        # 写入第一个文件（80%）
        with open(output_file1, 'w', encoding='utf-8') as f1:
            for idx in indices1:
                f1.write(lines[idx])

        # 写入第二个文件（20%）
        with open(output_file2, 'w', encoding='utf-8') as f2:
            for idx in indices2:
                f2.write(lines[idx])

        print(f"文件分割完成！")
        print(f"原始文件行数: {len(lines)}")
        print(f"文件1行数 ({split_ratio * 100}%): {len(indices1)}")
        print(f"文件2行数 ({100 - split_ratio * 100}%): {len(indices2)}")
        print(f"文件1保存至: {output_file1}")
        print(f"文件2保存至: {output_file2}")

    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_file}")
    except Exception as e:
        print(f"处理文件时发生错误: {e}")




# 使用示例
if __name__ == "__main__":
    # 配置参数
    input_filename = "文本分类练习.csv"  # 输入文件名
    output_filename1 = "training_data.csv"  # 80%的输出文件名
    output_filename2 = "validation_data.csv"  # 20%的输出文件名

    # 方法1：使用shuffle
    print("使用方法1分割文件:")
    split_file(input_filename, output_filename1, output_filename2, seed=42)

