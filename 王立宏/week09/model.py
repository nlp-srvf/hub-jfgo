# coding: utf-8

'''
model：定义模型结构
'''

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from typing import List, Tuple, Optional


class CRF(nn.Module):
    """
    条件随机场(CRF)层
    用于NER序列标注任务，处理标签之间的依赖关系
    """
    
    def __init__(self, num_tags: int):
        """
        初始化CRF层
        
        Args:
            num_tags: 标签数量
        """
        super(CRF, self).__init__()
        self.num_tags = num_tags
        
        # 转移矩阵：transitions[i][j] 表示从标签j转移到标签i的分数
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        
        # 开始和结束标签的索引
        self.start_tag = num_tags  # 开始标签
        self.end_tag = num_tags + 1   # 结束标签
        
        # 扩展转移矩阵，包含开始和结束标签
        # 实际使用时矩阵大小为 (num_tags + 2) x (num_tags + 2)
        self.transitions_with_start_end = nn.Parameter(
            torch.randn(num_tags + 2, num_tags + 2)
        )
        
        # 初始化参数
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化参数"""
        nn.init.uniform_(self.transitions, -0.01, 0.01)  # 减小初始化范围
        nn.init.uniform_(self.transitions_with_start_end, -0.01, 0.01)
        
        # 设置一些约束：从开始标签不能转移到结束标签
        self.transitions_with_start_end.data[self.end_tag, self.start_tag] = -10000
        # 设置其他约束逻辑...
    
    def _compute_forward(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        计算前向算法（分割函数）
        
        Args:
            emissions: 发射概率 [batch_size, seq_len, num_tags]
            mask: 掩码 [batch_size, seq_len]
            
        Returns:
            对数配分函数
        """
        batch_size, seq_len, num_tags = emissions.size()
        
        # 初始化前向变量
        # 从开始标签到第一个时间步的所有标签
        forward_var = emissions[:, 0] + self.transitions_with_start_end[self.start_tag, :num_tags]
        
        for i in range(1, seq_len):
            # 当前时间步的发射概率
            emit_score = emissions[:, i].unsqueeze(2).expand(batch_size, num_tags, num_tags)
            
            # 转移分数
            trans_score = self.transitions[:num_tags, :num_tags].unsqueeze(0).expand(batch_size, num_tags, num_tags)
            
            # 前向递推
            next_forward_var = forward_var.unsqueeze(1).expand(batch_size, num_tags, num_tags) + trans_score + emit_score
            
            # 使用log-sum-exp技巧避免数值溢出
            forward_var = torch.logsumexp(next_forward_var, dim=1)
            
            # 应用mask
            mask_i = mask[:, i].unsqueeze(1).expand(batch_size, num_tags)
            forward_var = forward_var * mask_i + forward_var * (~mask_i)
        
        # 添加到结束标签的转移
        forward_var = forward_var + self.transitions_with_start_end[:num_tags, self.end_tag]
        
        return torch.logsumexp(forward_var, dim=1)
    
    def _compute_score(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        计算给定标签序列的分数
        
        Args:
            emissions: 发射概率 [batch_size, seq_len, num_tags]
            tags: 标签序列 [batch_size, seq_len]
            mask: 掩码 [batch_size, seq_len]
            
        Returns:
            标签序列的分数
        """
        batch_size, seq_len, num_tags = emissions.size()
        score = torch.zeros(batch_size, device=emissions.device)
        
        # 添加从开始标签到第一个标签的转移分数
        score += self.transitions_with_start_end[self.start_tag, tags[:, 0]]
        
        for i in range(seq_len):
            # 添加发射分数
            score += emissions[torch.arange(batch_size), i, tags[:, i]] * mask[:, i]
            
            if i > 0:
                # 添加转移分数
                score += self.transitions[tags[:, i], tags[:, i-1]] * mask[:, i]
        
        # 添加到最后一个标签到结束标签的转移分数
        last_tag_indices = mask.sum(1) - 1
        last_tags = tags[torch.arange(batch_size), last_tag_indices]
        score += self.transitions_with_start_end[last_tags, self.end_tag]
        
        return score
    
    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        计算负对数似然损失
        
        Args:
            emissions: 发射概率 [batch_size, seq_len, num_tags]
            tags: 标签序列 [batch_size, seq_len]
            mask: 掩码 [batch_size, seq_len]
            
        Returns:
            负对数似然损失
        """
        forward_score = self._compute_forward(emissions, mask)
        gold_score = self._compute_score(emissions, tags, mask)
        
        return forward_score - gold_score
    
    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        """
        维特比解码，找到最优标签序列
        
        Args:
            emissions: 发射概率 [batch_size, seq_len, num_tags]
            mask: 掩码 [batch_size, seq_len]
            
        Returns:
            最优标签序列列表
        """
        batch_size, seq_len, num_tags = emissions.size()
        
        # 初始化维特比变量
        viterbi_vars = []
        viterbi_vars.append(emissions[:, 0] + self.transitions_with_start_end[self.start_tag, :num_tags])
        
        # 回溯指针
        backpointers = []
        
        for i in range(1, seq_len):
            # 计算当前时间步的维特比变量
            emit_score = emissions[:, i].unsqueeze(2).expand(batch_size, num_tags, num_tags)
            trans_score = self.transitions[:num_tags, :num_tags].unsqueeze(0).expand(batch_size, num_tags, num_tags)
            
            next_viterbi_var = viterbi_vars[-1].unsqueeze(1).expand(batch_size, num_tags, num_tags) + trans_score + emit_score
            
            # 找到最佳路径
            best_values, best_tags = torch.max(next_viterbi_var, dim=1)
            
            viterbi_vars.append(best_values)
            backpointers.append(best_tags)
        
        # 添加结束转移
        terminal_var = viterbi_vars[-1] + self.transitions_with_start_end[:num_tags, self.end_tag]
        best_values, best_tags = torch.max(terminal_var, dim=1)
        
        # 回溯找到最优路径
        path_scores = []
        best_paths = []
        
        for batch_idx in range(batch_size):
            path = [best_tags[batch_idx].item()]
            
            # 反向回溯
            for backpointer in reversed(backpointers):
                path.append(backpointer[batch_idx, path[-1]].item())
            
            # 反转路径
            path.reverse()
            
            # 根据mask截断路径
            seq_len_actual = mask[batch_idx].sum().item()
            best_paths.append(path[:seq_len_actual])
        
        return best_paths


class BertCRFForNER(BertPreTrainedModel):
    """
    BERT + CRF 的NER模型
    结合BERT的特征提取能力和CRF的序列建模能力
    """
    
    def __init__(self, config):
        """
        初始化BERT+CRF模型
        
        Args:
            config: 模型配置
        """
        super().__init__(config)
        
        # BERT层
        self.bert = BertModel(config)
        
        # dropout层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # 分类层
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # CRF层
        self.crf = CRF(config.num_labels)
        
        # 初始化权重
        self.post_init()
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                return_dict: bool = None) -> dict:
        """
        前向传播
        
        Args:
            input_ids: 输入token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            token_type_ids: token类型 IDs [batch_size, seq_len]
            labels: 标签 [batch_size, seq_len]
            return_dict: 是否返回字典格式
            
        Returns:
            包含损失和预测结果的字典
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # BERT特征提取
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict
        )
        
        # 获取序列表示
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]
        
        # Dropout
        sequence_output = self.dropout(sequence_output)
        
        # 分类层得到发射概率
        emissions = self.classifier(sequence_output)  # [batch_size, seq_len, num_labels]
        
        # 创建mask (忽略padding tokens)
        mask = attention_mask.bool()
        
        result = {
            'emissions': emissions,
            'mask': mask
        }
        
        if labels is not None:
            # 计算CRF损失
            # 注意：需要将-100转换为有效的mask
            labels_mask = (labels != -100)
            loss = self.crf(emissions, labels, labels_mask)
            result['loss'] = loss
        
        return result
    
    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        """
        解码得到最优标签序列
        
        Args:
            emissions: 发射概率 [batch_size, seq_len, num_labels]
            mask: 掩码 [batch_size, seq_len]
            
        Returns:
            最优标签序列列表
        """
        return self.crf.decode(emissions, mask)


def create_model(config: dict) -> BertCRFForNER:
    """
    创建NER模型
    
    Args:
        config: 配置字典
        
    Returns:
        初始化的模型
    """
    # 从预训练BERT模型加载配置
    from transformers import BertConfig
    
    bert_config = BertConfig.from_pretrained(config['bert_path'])
    bert_config.num_labels = config['class_num']
    bert_config.hidden_dropout_prob = 0.1
    
    # 创建模型
    model = BertCRFForNER.from_pretrained(
        config['bert_path'],
        config=bert_config
    )
    
    return model


def test_model():
    """
    测试模型功能
    """
    from config import Config
    
    print("🧪 测试BERT+CRF模型...")
    
    # 创建模型
    model = create_model(Config)
    
        # 创建测试数据
    batch_size, seq_len = Config['batch_size'], Config['max_length']
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    labels = torch.randint(0, Config['class_num'], (batch_size, seq_len), dtype=torch.long)
    
    # 测试前向传播
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
        
        print(f"📊 Emissions形状: {outputs['emissions'].shape}")
        print(f"💰 Loss值: {outputs['loss'].mean().item():.4f}")
        
        # 测试解码
        predictions = model.decode(outputs['emissions'], outputs['mask'])
        print(f"🎯 预测结果数量: {len(predictions)}")
        print(f"📏 第一个预测长度: {len(predictions[0])}")
    
    print("✅ 模型测试完成！")


if __name__ == "__main__":
    test_model()

