"""
ECG-R1 vLLM 插件

通过 vLLM 的插件系统注册 ECGR1ForConditionalGeneration 模型。

使用方式：
1. 在本仓库执行 `pip install -e .`，注册 pyproject.toml 中的 vLLM entry point。
2. 启动 `scripts/serve_rollout.sh`。
3. vLLM 启动时会自动加载插件。
"""

import os

_registered_processes = set()


def register():
    """
    vLLM 插件入口函数
    
    注册 ECGR1ForConditionalGeneration 到 vLLM ModelRegistry
    """
    pid = os.getpid()
    print(f"[ECG-R1 Plugin] register() called in process {pid}", flush=True)
    
    # 避免在同一进程中重复注册
    if pid in _registered_processes:
        print(f"[ECG-R1 Plugin] Already registered in process {pid}", flush=True)
        return
    
    try:
        from vllm.model_executor.models import ModelRegistry
        from .ecg_r1_model import ECGR1ForConditionalGeneration
        
        # 注册模型
        ModelRegistry.register_model(
            "ECGR1ForConditionalGeneration",
            ECGR1ForConditionalGeneration
        )
        
        _registered_processes.add(pid)
        print(f"✅ [ECG-R1 Plugin] ECGR1ForConditionalGeneration registered in process {pid}", flush=True)
        
    except Exception as e:
        print(f"❌ [ECG-R1 Plugin] Failed to register: {e}", flush=True)
        import traceback
        traceback.print_exc()


# 为了兼容性，也导出模型类
def get_model_class():
    """获取模型类（用于直接导入）"""
    from .ecg_r1_model import ECGR1ForConditionalGeneration
    return ECGR1ForConditionalGeneration
