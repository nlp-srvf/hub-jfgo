# 1. 安装miniconda
# 2. 配置环境
    conda activate -n py312 python=3.12
    conda activate py312
    pip install numpy==1.26.4 pandas==2.2.2 matplotlib==3.9.2 Scikit-learn==1.5.1 gensim peft==0.15.0 transformers==4.55.0 -i https://pypi.tuna.tsinghua.edu.cn/simple 
    安装pytorch-gpu报错，更换为conda安装
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
