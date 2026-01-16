
# model.py
from transformers import BertLMHeadModel, BertConfig

def get_model(config):
    """
    加载 BertLMHeadModel 并配置为解码器模式。
    BertLMHeadModel 在 BERT 输出层上加了一个 Linear 层用于预测词表中的下一个词。
    """
    print(f"Loading pre-trained BERT from {config.bert_path}...")

    # 关键步骤：设置 is_decoder=True
    # 这会激活因果注意力掩码（Causal Masking），防止模型看见“未来”的词
    model = BertLMHeadModel.from_pretrained(
        config.bert_path,
        is_decoder=True
    )

    model.to(config.device)
    return model