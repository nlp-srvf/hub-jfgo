# coding: utf-8

'''
config.py：输入模型配置参数，如学习率、模型保存位置等
'''
#!/usr/bin/env python3
"""
设备工具类 - 自动检测和选择最佳设备
支持GPU、NPU、MPS、CPU
"""

import torch
import platform
import warnings

def auto_select_device():
    """
    自动选择最佳设备
    
    优先级: NPU > GPU > MPS > CPU
    
    Returns:
        str: 设备类型 ('cuda', 'npu', 'mps', 'cpu')
    """
    
    # 1. 检查NPU (华为昇腾)
    if _check_npu_available():
        return 'npu'
    
    # 2. 检查CUDA (NVIDIA GPU)
    elif _check_cuda_available():
        return 'cuda'
    
    # 3. 检查MPS (Apple Silicon GPU)
    elif _check_mps_available():
        return 'mps'
    
    # 4. 默认使用CPU
    else:
        return 'cpu'

def _check_npu_available():
    """检查NPU是否可用"""
    try:
        import torch_npu
        if torch_npu.npu.is_available():
            device_count = torch_npu.npu.device_count()
            if device_count > 0:
                print(f"✅ 检测到 {device_count} 个NPU设备")
                return True
        return False
    except ImportError:
        return False
    except Exception as e:
        warnings.warn(f"检查NPU时出错: {e}")
        return False

def _check_cuda_available():
    """检查CUDA是否可用"""
    try:
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if device_count > 0:
                # 获取GPU名称
                gpu_name = torch.cuda.get_device_name(0)
                print(f"✅ 检测到 {device_count} 个CUDA设备: {gpu_name}")
                return True
        return False
    except Exception as e:
        warnings.warn(f"检查CUDA时出错: {e}")
        return False

def _check_mps_available():
    """检查MPS是否可用（Apple Silicon）"""
    try:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✅ 检测到MPS设备（Apple Silicon）")
            return True
        return False
    except Exception as e:
        warnings.warn(f"检查MPS时出错: {e}")
        return False

def get_device_info():
    """获取设备详细信息"""
    device = auto_select_device()
    info = {
        'selected_device': device,
        'device_count': 0,
        'device_name': None,
        'platform': platform.system(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__
    }
    
    if device == 'cuda':
        info['device_count'] = torch.cuda.device_count()
        info['device_name'] = torch.cuda.get_device_name(0)
        info['memory_total'] = torch.cuda.get_device_properties(0).total_memory
        info['memory_allocated'] = torch.cuda.memory_allocated()
        
    elif device == 'npu':
        try:
            import torch_npu
            info['device_count'] = torch_npu.npu.device_count()
            # NPU设备信息可能需要torch_npu提供的特定API
            info['device_name'] = 'NPU'
        except ImportError:
            pass
            
    elif device == 'mps':
        info['device_name'] = 'Apple Silicon GPU'
        
    else:
        info['device_name'] = 'CPU'
    
    return info

def print_device_info():
    """打印设备信息"""
    info = get_device_info()
    
    print("=" * 60)
    print("🖥️  设备信息")
    print("=" * 60)
    print(f"📱 平台: {info['platform']}")
    print(f"🐍 Python版本: {info['python_version']}")
    print(f"🔥 PyTorch版本: {info['torch_version']}")
    print(f"🎯 选择设备: {info['selected_device'].upper()}")
    print(f"🔧 设备名称: {info['device_name']}")
    print(f"📊 设备数量: {info['device_count']}")
    
    if 'memory_total' in info:
        memory_gb = info['memory_total'] / (1024**3)
        allocated_gb = info['memory_allocated'] / (1024**3)
        print(f"💾 总内存: {memory_gb:.2f} GB")
        print(f"💾 已用内存: {allocated_gb:.2f} GB")
    
    print("=" * 60)

def set_device(device=None):
    """
    设置并返回torch设备
    
    Args:
        device: 指定设备，如果为None则自动选择
        
    Returns:
        torch.device: torch设备对象
    """
    if device is None:
        device = auto_select_device()
    
    if device == 'cuda':
        torch_device = torch.device('cuda')
    elif device == 'npu':
        torch_device = torch.device('npu')
    elif device == 'mps':
        torch_device = torch.device('mps')
    else:
        torch_device = torch.device('cpu')
    
    return torch_device

def optimize_for_device(device=None):
    """
    针对不同设备进行优化设置
    
    Args:
        device: 设备类型
    """
    if device is None:
        device = auto_select_device()
    
    # CUDA优化
    if device == 'cuda':
        # 启用cudnn benchmark
        torch.backends.cudnn.benchmark = True
        # 启用cudnn deterministic（如果需要可重现结果）
        # torch.backends.cudnn.deterministic = True
        print("🚀 启用CUDA优化设置")
    
    # MPS优化
    elif device == 'mps':
        # MPS的一些优化设置
        print("🚀 启用MPS优化设置")
    
    # NPU优化
    elif device == 'npu':
        try:
            import torch_npu
            # NPU特定优化设置
            torch_npu.npu.set_compile_mode(jit_compile=False)
            print("🚀 启用NPU优化设置")
        except ImportError:
            pass
    
    # CPU优化
    else:
        # 设置CPU线程数
        torch.set_num_threads(4)
        print("🚀 启用CPU优化设置")

# 测试函数
def test_device():
    """测试设备功能"""
    device = set_device()
    print(f"🧪 测试设备: {device}")
    
    # 创建测试张量
    try:
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        z = torch.mm(x, y)
        print(f"✅ 设备测试成功，张量形状: {z.shape}")
        return True
    except Exception as e:
        print(f"❌ 设备测试失败: {e}")
        return False

if __name__ == "__main__":
    # 打印设备信息
    print_device_info()
    
    # 测试设备
    test_device()
    
    # 优化设置
    device = auto_select_device()
    optimize_for_device(device)