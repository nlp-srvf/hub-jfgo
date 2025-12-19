# coding: utf-8

'''
main.py：使用训练数据集完成文本分类实验
'''

import pandas as pd

from config import Config
from loader import DataLoaderProcessor
from model_bert import BertClassifier
from model_fast_text import FastTextClassifier
from model_text_cnn import TextCNNClassifier
from model_lstm import LSTMClassifier
from model_gated_cnn import GatedCNNClassifier
from model_lr import NaiveBayesClassifier
from model_svm import SVMClassifier
from evaluate import evaluate_model, print_evaluation_results, repeat_evaluation

def train_and_evaluate_model(model_name, config, data_processor, repeat_times=10):
    """训练并评估单个模型，支持多次重复评估"""
    print(f"\n{'='*60}")
    print(f"开始训练和评估 {model_name} 模型")
    print(f"{'='*60}")
    print(f"将进行 {repeat_times} 次重复评估")

    # 模型类映射
    model_class_map = {
        'BERT': BertClassifier,
        'FastText': FastTextClassifier,
        'TextCNN': TextCNNClassifier,
        'LSTM': LSTMClassifier,
        'GatedCNN': GatedCNNClassifier,
        'NaiveBayes': NaiveBayesClassifier,
        'SVM': SVMClassifier
    }

    # 加载数据
    X, y = data_processor.load_all_data()

    # 根据模型类型选择字典构建函数
    if model_name in ['TextCNN', 'LSTM', 'GatedCNN']:
        vocab_builder = data_processor.build_vocab
    else:
        vocab_builder = None

    # 进行多次重复评估
    results = repeat_evaluation(
        model_class=model_class_map[model_name],
        config=config,
        X=X,
        y=y,
        model_name=model_name,
        vocab_builder=vocab_builder,
        n=repeat_times,
        data_processor=data_processor
    )

    return results

def main():
    """主函数"""
    # 初始化配置
    config = Config()

    # 读取重复评估次数（默认10次）
    repeat_times = getattr(config, 'repeat_evaluation_times', 10)

    print(f"\n{'='*80}")
    print(f"将每个模型进行 {repeat_times} 次重复评估")
    print(f"{'='*80}")

    # 数据处理器
    data_processor = DataLoaderProcessor(config)

    # 定义要训练的模型
    models_to_train = [
        'NaiveBayes',
        'SVM',
        'TextCNN',
        'LSTM',
        'GatedCNN',
        'FastText',
        'BERT'
    ]

    # 保存所有详细的评估结果
    all_detailed_results = []
    # 保存统计摘要
    all_summary_results = []

    # 训练和评估每个模型
    for model_name in models_to_train:
        try:
            result = train_and_evaluate_model(model_name, config, data_processor, repeat_times=repeat_times)

            if result and 'all_results' in result:
                # 保存详细的每次评估结果
                all_detailed_results.extend(result['all_results'])

                # 保存统计摘要
                if 'statistics' in result and result['statistics']:
                    all_summary_results.append(result['statistics'])
        except Exception as e:
            print(f"模型 {model_name} 评估失败: {str(e)}")
            continue

    # 保存所有详细结果到CSV文件
    if all_detailed_results:
        # 根据eval_path生成详细结果文件名
        detailed_path = config.result_path.replace('.csv', '_detailed.csv')
        df_detailed = pd.DataFrame(all_detailed_results)
        df_detailed.to_csv(detailed_path, index=False)
        print(f"\n详细评估结果已保存到 {detailed_path}")
        print(f"包含 {len(all_detailed_results)} 条评估记录")

        # 显示前几行数据示例
        print("\n详细结果示例（前5行）：")
        print(df_detailed.head())

    # 保存统计摘要到主文件
    if all_summary_results:
        df_summary = pd.DataFrame(all_summary_results)
        df_summary.to_csv(config.result_path, index=False)
        print(f"\n统计摘要结果已保存到 {config.result_path}")

        # 显示汇总结果
        print(f"\n{'='*80}")
        print("模型性能统计摘要:")
        print(f"{'='*80}")

        # 显示关键统计信息
        display_cols = ['model', 'accuracy_mean', 'accuracy_std', 'f1_mean', 'f1_std',
                       'precision_mean', 'recall_mean', 'evaluation_count']
        print(df_summary[display_cols])
        print(f"{'='*80}")
    else:
        print("没有任何模型成功训练！")

if __name__ == "__main__":
    main()

