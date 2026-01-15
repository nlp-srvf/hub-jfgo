| 模型 | 位置编码 | transformer结构 | 多头机制 | ff层设计 | 归一化层选择 | 激活函数 | 是否使用bias |
|------|----------|----------------|----------|----------|--------------|----------|--------------|
| baichuan2-7b | RoPE | 串行 | 传统方式 | gated形式 | RMSnorm / pre norm | SiLU | 无bias |
| baichuan2-13b | Alibi | 串行 | 传统方式 | gated形式 | RMSnorm / pre norm | SiLU | 无bias |
| chatglm2 | RoPE | 串行 | grouped-query | 传统方式 | RMSnorm / pre norm | SiLU | qkv有bias，其他线性层无bias |
| chatglm3 | RoPE | 串行 | grouped-query | 传统方式 | RMSnorm / pre norm | SiLU | qkv有bias，其他线性层无bias |
| deepseek | RoPE | 串行 | 传统方式 | MOE, gated形式 | RMSnorm / pre norm | Sigmoid, SiLU | 无bias |
| gemma | RoPE | 串行 | grouped-query | 传统方式 | RMSnorm / pre norm | gelu | 无bias |
| grok1 | RoPE | 串行 | grouped-query | MOE | RMSnorm / Sandwich norm | softmax, gelu | 无bias |
| llama | RoPE | 串行 | grouped-query | gated形式 | RMSnorm / pre norm | SiLU | 无bias |
| Mixtral | RoPE | 串行 | grouped-query | MOE, gated形式 | RMSnorm / pre norm | softmax, SiLU | 无bias |
| moss | Sinusoidal | 平行 | 传统方式 | 传统方式 | LayerNorm | gelu_new | sa无bias, ff有bias |
| qwen | RoPE | 串行 | 传统方式 | gated形式 | RMSnorm / pre norm | SiLU | qkv有bias，其他线性层无bias |
