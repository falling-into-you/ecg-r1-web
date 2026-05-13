"""
ECG-R1 vLLM 多模态处理器

步骤 2 & 3：
- ECGR1ProcessingInfo: 添加 ECG 模态限制
- ECGR1DummyInputsBuilder: 生成 ECG 虚拟输入
- ECGR1MultiModalProcessor: 处理 ECG 数据

用法：
    这些类会通过 @MULTIMODAL_REGISTRY.register_processor 注册到模型上
"""

import os
import sys
from typing import Any, Mapping, Optional, Sequence

import torch
import numpy as np

# 添加项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from vllm.multimodal import MultiModalDataDict
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import (
    MultiModalDataItems, 
    MultiModalDataParser,
    ModalityDataItems,
    # 注意：不导入 EmbeddingItems！ECGDataItems 继承 ModalityDataItems
)
from vllm.multimodal.processing import PromptUpdate, PromptReplacement
from transformers import BatchFeature

# 导入父类
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLProcessingInfo,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
)


# ==================== ECG 数据解析器 ====================

class ECGDataItems(ModalityDataItems):
    """
    ECG 数据项
    
    ⚠️ 重要：不继承 EmbeddingItems！
    
    原因：vLLM 的 _hf_processor_applies_updates() 会检查 mm_items 中是否有 EmbeddingItems。
    如果有，返回 False，导致使用空参数处理 prompt，图像 token 数量错误。
    
    通过继承 ModalityDataItems 而非 EmbeddingItems，确保 _hf_processor_applies_updates() 
    返回 True，让 HF Processor 使用正确的 mm_processor_kwargs 处理图像。
    """
    
    def __init__(self, data: torch.Tensor):
        # data shape: (batch, leads, samples) 或 (batch, tokens, hidden)
        self.data = data
        self._modality = "ecg"
    
    @property
    def modality(self) -> str:
        return self._modality
    
    def get_count(self) -> int:
        """返回 ECG 数据项数量"""
        return self.data.shape[0] if self.data.ndim >= 1 else 1
    
    def get(self, index: int) -> torch.Tensor:
        """获取指定索引的 ECG 数据"""
        return self.data[index]
    
    def get_processor_data(self) -> Mapping[str, Any]:
        """返回需要 HF Processor 处理的数据 - ECG 不需要"""
        return {}
    
    def get_passthrough_data(self) -> Mapping[str, Any]:
        """返回需要透传到模型的数据"""
        return {"ecg_embeds": self.data}


class ECGR1DataParser(MultiModalDataParser):
    """
    ECG-R1 数据解析器
    
    扩展 MultiModalDataParser，添加 ECG 模态支持
    """
    
    def _parse_ecg_data(self, data) -> Optional[ModalityDataItems]:
        """解析 ECG 数据"""
        if data is None:
            return None
        
        # 空数据检查
        if isinstance(data, (list, tuple)) and len(data) == 0:
            return None
        
        # 如果已经是 tensor，转换为 ECGDataItems
        if isinstance(data, torch.Tensor):
            # 确保是 3D: (batch, leads, samples) 或 (batch, tokens, hidden)
            if data.ndim == 2:
                data = data.unsqueeze(0)
            return ECGDataItems(data)
        
        # 如果是 list of tensors
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], torch.Tensor):
                # Stack tensors
                stacked = torch.stack([t if t.dim() == 2 else t.squeeze(0) for t in data])
                return ECGDataItems(stacked)
        
        # 其他情况，尝试转换为 tensor
        try:
            tensor_data = torch.tensor(data)
            if tensor_data.ndim == 2:
                tensor_data = tensor_data.unsqueeze(0)
            return ECGDataItems(tensor_data)
        except Exception:
            return None
    
    def _get_subparsers(self) -> Mapping[str, Any]:
        """扩展子解析器，添加 ECG 支持"""
        subparsers = dict(super()._get_subparsers())
        subparsers["ecg"] = self._parse_ecg_data
        return subparsers


# ==================== ECG 常量 ====================

ECG_PLACEHOLDER = "<|ecg_pad|>"
ECG_START_TOKEN = "<|ecg_start|>"
ECG_END_TOKEN = "<|ecg_end|>"

def get_env_args(key: str, dtype: type, default: Any) -> Any:
    """从环境变量获取配置"""
    val = os.environ.get(key)
    if val is None:
        return default
    if dtype == bool:
        return val.lower() in ('true', '1', 'yes')
    return dtype(val)


# ==================== 步骤 3A: ProcessingInfo ====================

class ECGR1ProcessingInfo(Qwen3VLProcessingInfo):
    """
    ECG-R1 处理信息类
    
    扩展 Qwen3VLProcessingInfo，添加 ECG 模态支持
    """
    
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        """返回每种模态的数量限制"""
        limits = super().get_supported_mm_limits()
        # 添加 ECG 模态限制：每个请求最多 1 个 ECG
        limits["ecg"] = 1
        return limits
    
    def get_ecg_num_tokens(self) -> int:
        """获取 ECG 模态的 token 数量"""
        ecg_seq_length = get_env_args('ECG_SEQ_LENGTH', int, 5000)
        ecg_patch_size = get_env_args('ECG_PATCH_SIZE', int, 50)
        # tokens = seq_length / patch_size + 1 (cls token)
        return ecg_seq_length // ecg_patch_size + 1


# ==================== 步骤 3B: DummyInputsBuilder ====================

class ECGR1DummyInputsBuilder(Qwen3VLDummyInputsBuilder):
    """
    ECG-R1 虚拟输入构建器
    
    用于 vLLM 的内存预分配和性能分析
    """
    
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        """生成包含 ECG placeholder 的虚拟文本"""
        text = super().get_dummy_text(mm_counts)
        
        num_ecgs = mm_counts.get("ecg", 0)
        if num_ecgs > 0:
            ecg_token = f"{ECG_START_TOKEN}{ECG_PLACEHOLDER}{ECG_END_TOKEN}"
            text = text + ecg_token * num_ecgs
        
        return text
    
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        """生成虚拟多模态数据"""
        data = super().get_dummy_mm_data(seq_len, mm_counts)
        
        num_ecgs = mm_counts.get("ecg", 0)
        if num_ecgs > 0:
            # 生成虚拟 ECG 数据: (batch, 12 leads, 5000 samples)
            ecg_seq_length = get_env_args('ECG_SEQ_LENGTH', int, 5000)
            data["ecg"] = [torch.zeros(12, ecg_seq_length) for _ in range(num_ecgs)]
        
        return data


# ==================== 步骤 2: MultiModalProcessor ====================

class ECGR1MultiModalProcessor(Qwen3VLMultiModalProcessor):
    """
    ECG-R1 多模态处理器
    
    继承 Qwen3VLMultiModalProcessor，添加 ECG 数据处理
    
    ⚠️ 重要修改：重写 _hf_processor_applies_updates()
    当有 ECG 数据时，返回 False，让 vLLM 调用 _apply_prompt_updates() 处理所有 placeholder。
    同时保存 mm_kwargs 用于后续处理。
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 保存 mm_kwargs 用于 _apply_hf_processor_text_only
        self._saved_mm_kwargs: Mapping[str, object] = {}
    
    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        """
        判断 HF Processor 是否已经应用了 prompt updates
        
        ⚠️ 关键改动：始终返回 True
        
        原因：
        1. 当返回 False 时，vLLM 会调用 _apply_prompt_updates 替换 placeholder
        2. 但 HF Processor 已经在 _call_hf_processor 中展开了图像 placeholder
        3. _apply_prompt_updates 的 text-based 替换会导致 token 数量错误
        
        解决方案：返回 True，让 vLLM 使用 _find_mm_placeholders 查找已展开的 placeholder
        ECG 的 placeholder 将在 _call_hf_processor 中手动展开
        """
        return True
    
    def _get_data_parser(self) -> MultiModalDataParser:
        """返回支持 ECG 的数据解析器"""
        # 使用自定义的 ECGR1DataParser，支持 ECG 模态
        return ECGR1DataParser(video_needs_metadata=True)
    
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """
        调用 HuggingFace processor，并处理 ECG 数据
        
        ⚠️ 重要：手动展开 ECG placeholder
        
        因为 _hf_processor_applies_updates 返回 True，vLLM 期望所有 placeholder 都已展开。
        HF Processor 会展开图像 placeholder，但不认识 ECG placeholder。
        所以我们需要在这里手动展开 ECG placeholder。
        
        注意：ECG 数据不在 mm_data 中（ECGDataItems.get_processor_data() 返回 {}），
        而是通过 passthrough_data 传递。我们只需要展开 prompt 中的 ECG placeholder。
        """
        mm_data = dict(mm_data)
        
        # 计算 ECG token 数量
        ecg_seq_length = get_env_args('ECG_SEQ_LENGTH', int, 5000)
        ecg_patch_size = get_env_args('ECG_PATCH_SIZE', int, 50)
        tokens_per_ecg = ecg_seq_length // ecg_patch_size + 1  # +1 for cls token
        
        # 检查 prompt 中是否有 ECG placeholder，如果有则展开
        ecg_pattern = f"{ECG_START_TOKEN}{ECG_PLACEHOLDER}{ECG_END_TOKEN}"
        has_ecg_placeholder = ecg_pattern in prompt
        
        if has_ecg_placeholder:
            # 展开 ECG placeholder: <|ecg_pad|> -> <|ecg_pad|> * tokens_per_ecg
            ecg_expanded = f"{ECG_START_TOKEN}" + ECG_PLACEHOLDER * tokens_per_ecg + f"{ECG_END_TOKEN}"
            
            # 替换所有 ECG placeholder
            while ecg_pattern in prompt:
                prompt = prompt.replace(ecg_pattern, ecg_expanded, 1)
        
        # 调用父类处理 image/video（prompt 已包含展开的 ECG placeholder）
        processed = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )
        
        return processed
    
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """获取多模态字段配置，添加 ECG 字段"""
        config = super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs)
        config = dict(config)  # 转为可变字典
        
        # 添加 ECG 字段配置
        if "ecg_embeds" in hf_inputs:
            config["ecg_embeds"] = MultiModalFieldConfig.batched("ecg")
        
        return config
    
    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        """获取 prompt 更新规则，添加 ECG placeholder 处理"""
        updates = list(super()._get_prompt_updates(
            mm_items, hf_processor_mm_kwargs, out_mm_kwargs
        ))
        
        # 获取 ECG token 数量
        ecg_seq_length = get_env_args('ECG_SEQ_LENGTH', int, 5000)
        ecg_patch_size = get_env_args('ECG_PATCH_SIZE', int, 50)
        tokens_per_ecg = ecg_seq_length // ecg_patch_size + 1  # +1 for cls token
        
        # 获取 tokenizer
        tokenizer = self.info.get_tokenizer()
        ecg_token_id = tokenizer.convert_tokens_to_ids(ECG_PLACEHOLDER)
        
        def get_ecg_replacement(item_idx: int):
            """替换单个 ECG placeholder 为多个 token"""
            return [ecg_token_id] * tokens_per_ecg
        
        # 添加 ECG placeholder 替换规则
        # 注意：因为我们在 _call_hf_processor 中已经展开了 ECG placeholder，
        # target 需要匹配展开后的形式
        ecg_expanded_pattern = f"{ECG_START_TOKEN}" + ECG_PLACEHOLDER * tokens_per_ecg + f"{ECG_END_TOKEN}"
        
        updates.append(
            PromptReplacement(
                modality="ecg",
                target=ecg_expanded_pattern,
                replacement=get_ecg_replacement,
            )
        )
        
        return updates
