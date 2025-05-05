# utils/torch_utils.py

import torch
import numpy as np
import random
import os

def set_seed(seed: int):
    """
    设置随机种子以保证实验可复现性。

    Args:
        seed (int): 要设置的随机种子。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 如果使用 CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # 为所有 GPU 设置种子 (如果有多卡)
        # 以下两项可能会影响性能，但有助于复现确定性卷积算法
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    print(f"随机种子已设置为: {seed}")

def get_device(device_arg: str = 'auto') -> torch.device:
    """
    根据输入参数或自动检测来获取计算设备 (CPU or CUDA)。

    Args:
        device_arg (str, optional): 设备参数。
            可以是 'auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1' 等。
            'auto' 会优先选择 CUDA (如果可用)，否则选择 CPU。
            默认为 'auto'。

    Returns:
        torch.device: 选择的计算设备对象。
    """
    if device_arg.lower() == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"自动选择设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    elif 'cuda' in device_arg.lower():
        if not torch.cuda.is_available():
            print(f"警告: 请求使用 CUDA ({device_arg})，但 CUDA 不可用。将使用 CPU。")
            device = torch.device("cpu")
        else:
            try:
                device = torch.device(device_arg)
                # 尝试访问设备以确保其有效
                _ = torch.tensor([1]).to(device)
                print(f"使用指定 CUDA 设备: {device_arg}")
            except (RuntimeError, AssertionError) as e:
                print(f"警告: 指定的 CUDA 设备 '{device_arg}' 无效或不可用: {e}。将使用默认 CUDA 设备 cuda:0。")
                try:
                    device = torch.device("cuda:0")
                    _ = torch.tensor([1]).to(device) # 再次检查默认设备
                except (RuntimeError, AssertionError):
                     print(f"警告: 默认 CUDA 设备 cuda:0 也无法使用。将使用 CPU。")
                     device = torch.device("cpu")

    else: # 假设是 'cpu'
        device = torch.device("cpu")
        print("使用 CPU 设备。")

    # 打印 CUDA 相关信息 (如果使用 CUDA)
    if device.type == 'cuda':
        print(f"  CUDA 设备名称: {torch.cuda.get_device_name(device)}")
        print(f"  CUDA 可用内存: {torch.cuda.get_device_properties(device).total_memory / (1024**3):.2f} GB")

    return device

# --- 示例用法 ---
if __name__ == '__main__':
    print("测试 torch_utils.py...")

    # 测试设置种子
    print("\n测试设置种子:")
    set_seed(12345)
    print("PyTorch 随机数:", torch.rand(1).item())
    print("NumPy 随机数:", np.random.rand())
    print("Python random 随机数:", random.random())
    # 多次调用 set_seed(相同值) 应该导致后续随机数生成相同
    set_seed(12345)
    print("再次设置种子后 PyTorch 随机数:", torch.rand(1).item())

    # 测试获取设备
    print("\n测试获取设备:")
    device_auto = get_device('auto')
    print(f"  自动选择结果: {device_auto}")

    device_cpu = get_device('cpu')
    print(f"  指定 CPU 结果: {device_cpu}")

    # 测试 CUDA (如果可用)
    if torch.cuda.is_available():
        device_cuda0 = get_device('cuda:0')
        print(f"  指定 cuda:0 结果: {device_cuda0}")
        device_cuda_default = get_device('cuda')
        print(f"  指定 cuda 结果: {device_cuda_default}")
        # 测试无效设备
        device_invalid = get_device('cuda:99') # 假设你没有 99 号 GPU
        print(f"  指定无效 cuda:99 结果: {device_invalid}")
    else:
        print("\nCUDA 不可用，跳过 CUDA 设备测试。")
        device_cuda_unavailable = get_device('cuda:0')
        print(f"  在 CUDA 不可用时指定 cuda:0 结果: {device_cuda_unavailable}")