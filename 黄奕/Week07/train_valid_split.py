import random

def split_file(file_path):
    with open(file_path,'r',encoding='utf-8') as f:
        lines = f.readlines()[1:]

    random.shuffle(lines)
    num_train = int(len(lines) * 0.8)

    train_lines = lines[:num_train]
    valid_lines = lines[num_train:]

    with open('train_data.txt','w',encoding='utf-8') as f:
        f.writelines(train_lines)

    with open('valid_data.txt','w',encoding='utf-8') as f:
        f.writelines(valid_lines)


split_file('文本分类练习.csv')
