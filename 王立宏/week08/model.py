# coding: utf-8

'''
model：定义模型结构
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class TextEncoder(nn.Module):
    """文本编码器"""
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1, dropout=0.1):
        super(TextEncoder, self).__init__()
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        
        # LSTM层
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 输出层
        self.output_dim = hidden_size * 2  # 双向LSTM
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, lengths=None):
        """
        Args:
            x: 输入张量 (batch_size, seq_len)
            lengths: 实际长度 (batch_size,)
        Returns:
            encoded: 编码后的表示 (batch_size, output_dim)
        """
        # 词嵌入
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(embedded)
        
        # LSTM编码
        if lengths is not None:
            # 打包处理变长序列
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, 
                lengths, 
                batch_first=True, 
                enforce_sorted=False
            )
            lstm_out, (hidden, cell) = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # 使用最后一个时间步的输出
        # 对于双向LSTM，连接前向和后向的最后一个隐藏状态
        forward_hidden = hidden[-2]  # 前向最后一层
        backward_hidden = hidden[-1]  # 后向最后一层
        encoded = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        encoded = self.dropout(encoded)
        
        return encoded

class TripletTextMatchingModel(nn.Module):
    """三元组损失文本匹配模型"""
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=64, num_layers=1, dropout=0.1):
        super(TripletTextMatchingModel, self).__init__()
        
        # 文本编码器（共享参数）
        self.encoder = TextEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
    def forward(self, anchor, positive, negative, anchor_lengths=None, positive_lengths=None, negative_lengths=None):
        """
        前向传播
        Args:
            anchor: 锚点文本 (batch_size, seq_len)
            positive: 正样本文本 (batch_size, seq_len)  
            negative: 负样本文本 (batch_size, seq_len)
            anchor_lengths: 锚点文本长度 (batch_size,)
            positive_lengths: 正样本文本长度 (batch_size,)
            negative_lengths: 负样本文本长度 (batch_size,)
        Returns:
            anchor_emb, positive_emb, negative_emb: 编码后的表示
        """
        # 编码三个文本
        anchor_emb = self.encoder(anchor, anchor_lengths)
        positive_emb = self.encoder(positive, positive_lengths)
        negative_emb = self.encoder(negative, negative_lengths)
        
        return anchor_emb, positive_emb, negative_emb
    
    def encode(self, text, lengths=None):
        """
        编码单个文本
        Args:
            text: 输入文本 (batch_size, seq_len)
            lengths: 文本长度 (batch_size,)
        Returns:
            encoded: 编码后的表示 (batch_size, output_dim)
        """
        return self.encoder(text, lengths)

class TripletLoss(nn.Module):
    """三元组损失函数"""
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        计算三元组损失
        Args:
            anchor: 锚点表示 (batch_size, embed_dim)
            positive: 正样本表示 (batch_size, embed_dim)
            negative: 负样本表示 (batch_size, embed_dim)
        Returns:
            loss: 三元组损失
        """
        # 计算距离
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        # 计算损失
        loss = torch.mean(F.relu(pos_dist - neg_dist + self.margin))
        
        return loss

class CosineSimilarityLoss(nn.Module):
    """余弦相似度损失"""
    def __init__(self, margin=0.1):
        super(CosineSimilarityLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        计算余弦相似度损失
        Args:
            anchor: 锚点表示 (batch_size, embed_dim)
            positive: 正样本表示 (batch_size, embed_dim)
            negative: 负样本表示 (batch_size, embed_dim)
        Returns:
            loss: 余弦相似度损失
        """
        # 计算余弦相似度
        pos_sim = F.cosine_similarity(anchor, positive, dim=1)
        neg_sim = F.cosine_similarity(anchor, negative, dim=1)
        
        # 计算损失：希望正样本相似度高，负样本相似度低
        loss = torch.mean(F.relu(neg_sim - pos_sim + self.margin))
        
        return loss

def get_model(vocab_size, config):
    """获取模型实例"""
    model = TripletTextMatchingModel(
        vocab_size=vocab_size,
        embedding_dim=config['hidden_size'],
        hidden_size=config['hidden_size'] // 2,  # 因为双向LSTM会翻倍
        dropout=0.1
    )
    
    # 使用三元组损失
    criterion = TripletLoss(margin=1.0)
    
    return model, criterion

if __name__ == "__main__":
    # 测试代码
    from config import Config
    
    # 假设词汇表大小
    vocab_size = 5000
    batch_size = 32
    seq_len = 20
    
    # 创建模型
    model, criterion = get_model(vocab_size, Config)
    
    # 创建随机数据
    anchor = torch.randint(0, vocab_size, (batch_size, seq_len))
    positive = torch.randint(0, vocab_size, (batch_size, seq_len))
    negative = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 前向传播
    anchor_emb, positive_emb, negative_emb = model(anchor, positive, negative)
    
    print(f"Anchor embedding shape: {anchor_emb.shape}")
    print(f"Positive embedding shape: {positive_emb.shape}")
    print(f"Negative embedding shape: {negative_emb.shape}")
    
    # 计算损失
    loss = criterion(anchor_emb, positive_emb, negative_emb)
    print(f"Triplet loss: {loss.item():.4f}")
    
    # 测试编码单个文本
    text = torch.randint(0, vocab_size, (batch_size, seq_len))
    encoded = model.encode(text)
    print(f"Single text encoding shape: {encoded.shape}")
