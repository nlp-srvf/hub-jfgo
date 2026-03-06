import pandas as pd
from pathlib import Path

# 读取Excel文件
excel_path = Path('scenario/slot_fitting_templet.xlsx')

try:
    # 读取所有工作表
    xl = pd.ExcelFile(excel_path)
    print(f"Excel文件包含工作表: {xl.sheet_names}")
    
    # 读取第一个工作表
    df = pd.read_excel(excel_path)
    print("\n工作表内容:")
    print(df)
    
    # 检查列名
    print("\n列名:")
    print(df.columns.tolist())
    
    # 检查每一行的数据
    print("\n每行数据:")
    for index, row in df.iterrows():
        print(f"行 {index+1}: {row.to_dict()}")
        
except Exception as e:
    print(f"读取Excel文件失败: {e}")
