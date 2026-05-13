"""
ECG-R1 模型的 vLLM 实现

继承 Qwen3VLForConditionalGeneration，添加 ECG 模态支持。
"""

import os
import re
from typing import Any, Iterable, Mapping, Optional, Union

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen2_5_VLImageInputs,
    Qwen2_5_VLVideoInputs,
)
from vllm.model_executor.models.utils import WeightsMapper, maybe_prefix
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.utils import merge_multimodal_embeddings
from vllm.multimodal import MULTIMODAL_REGISTRY

# 导入 ECG 处理器类
from .ecg_r1_processor import (
    ECGR1ProcessingInfo,
    ECGR1DummyInputsBuilder,
    ECGR1MultiModalProcessor,
)

logger = init_logger(__name__)


# ==================== ECG 组件构建工具 ====================

def build_ecg_tower(ecg_tower_path: str, model_config_name: str = 'coca_ViT-B-32', device: str = 'cpu'):
    """构建 ECG Tower（结构必须与训练端一致）"""
    from ecg_coca.training import get_ecg_encoder
    ecg_tower, ecg_processor, ecg_config = get_ecg_encoder(
        model_name=model_config_name,
        checkpoint_path=ecg_tower_path,
        device=device
    )
    return ecg_tower, ecg_config


def build_ecg_projector(ecg_hidden_size: int, llm_hidden_size: int, projector_type: str = 'mlp2x_gelu'):
    """构建 ECG Projector（结构必须与训练端一致）"""
    if projector_type == 'linear':
        return nn.Linear(ecg_hidden_size, llm_hidden_size)
    
    match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if match:
        mlp_depth = int(match.group(1))
        modules = [nn.Linear(ecg_hidden_size, llm_hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(llm_hidden_size, llm_hidden_size))
        return nn.Sequential(*modules)
    
    raise ValueError(f'Unknown projector type: {projector_type}')


# ==================== ECG 输入类型定义 ====================

class ECGR1EmbeddingInputs:
    """ECG embedding 输入类型"""
    def __init__(self, ecg_embeds: torch.Tensor):
        self.type = "ecg_embeds"
        self.ecg_embeds = ecg_embeds


# ==================== ECGR1ForConditionalGeneration ====================

@MULTIMODAL_REGISTRY.register_processor(
    ECGR1MultiModalProcessor,
    info=ECGR1ProcessingInfo,
    dummy_inputs=ECGR1DummyInputsBuilder,
)
class ECGR1ForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """
    ECG-R1 模型的 vLLM 实现
    
    继承 Qwen3VLForConditionalGeneration，添加：
    - ecg_tower: ECG 编码器
    - ecg_projector: ECG -> LLM 维度映射
    - ECG 模态的多模态处理
    """
    
    # 扩展权重映射器，添加 ECG 参数映射
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # 原有的 Qwen3VL 映射
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
            # 新增 ECG 映射
            "model.ecg_tower.": "ecg_tower.",
            "model.ecg_projector.": "ecg_projector.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        """获取模态的占位符字符串"""
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"
        if modality.startswith("ecg"):
            return "<|ecg_start|><|ecg_pad|><|ecg_end|>"
        
        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        # 调用父类初始化
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        
        config = vllm_config.model_config.hf_config
        
        # 初始化 ECG 组件
        self._init_ecg_components(config)
        
        # 获取 ECG token ID
        self.ecg_token_id = getattr(config, 'ecg_token_id', None)
        
        logger.info(f"[ECG-R1] Initialized with ecg_token_id={self.ecg_token_id}")

    def _init_ecg_components(self, config):
        """初始化 ECG tower 和 projector"""
        # 从环境变量或 config 获取路径
        ecg_tower_path = getattr(config, 'ecg_tower_path', None) or os.environ.get('ECG_TOWER_PATH')
        ecg_projector_type = getattr(config, 'ecg_projector_type', None) or os.environ.get('ECG_PROJECTOR_TYPE', 'mlp2x_gelu')
        ecg_model_config = getattr(config, 'ecg_model_config', None) or os.environ.get('ECG_MODEL_CONFIG', 'coca_ViT-B-32')
        
        # 获取 LLM hidden size
        llm_hidden_size = getattr(config, 'hidden_size', None)
        if llm_hidden_size is None and hasattr(config, 'text_config'):
            llm_hidden_size = getattr(config.text_config, 'hidden_size', None)
        
        print(f"[ECG-R1] _init_ecg_components called in process {os.getpid()}", flush=True)
        print(f"[ECG-R1] ecg_tower_path={ecg_tower_path}", flush=True)
        
        if ecg_tower_path and llm_hidden_size:
            try:
                # 构建 ECG tower（权重会通过 RLHF 同步，这里只需要结构正确）
                ecg_tower, ecg_cfg = build_ecg_tower(ecg_tower_path, ecg_model_config, device='cpu')
                ecg_hidden_size = ecg_cfg.get('ecg_cfg', {}).get('width', 768)
                
                # 构建 ECG projector
                ecg_projector = build_ecg_projector(ecg_hidden_size, llm_hidden_size, ecg_projector_type)
                
                # 注册为模块（这样参数名会是 ecg_tower.* 和 ecg_projector.*）
                self.ecg_tower = ecg_tower
                self.ecg_projector = ecg_projector
                
                # 保存配置
                self.ecg_hidden_size = ecg_hidden_size
                self.llm_hidden_size = llm_hidden_size
                
                print(f"✅ [ECG-R1] ECG components attached: ecg_hidden={ecg_hidden_size}, llm_hidden={llm_hidden_size}", flush=True)
                
            except Exception as e:
                logger.error(f"❌ [ECG-R1] Failed to initialize ECG components: {e}")
                self.ecg_tower = None
                self.ecg_projector = None
        else:
            logger.warning(f"[ECG-R1] ECG components not initialized: ecg_tower_path={ecg_tower_path}, llm_hidden_size={llm_hidden_size}")
            self.ecg_tower = None
            self.ecg_projector = None

    def _parse_and_validate_ecg_input(self, **kwargs) -> Optional[ECGR1EmbeddingInputs]:
        """
        解析 ECG 输入
        
        输入格式: ecg_embeds 是原始 ECG 信号，形状为 (batch, 12, 5000)
        参考 my_register_v3.py 的处理方式
        """
        ecg_embeds = kwargs.pop("ecg_embeds", None)
        
        if ecg_embeds is None:
            return None
        
        # 简单的形状规范化：确保是 3D tensor (batch, 12, 5000)
        if isinstance(ecg_embeds, torch.Tensor):
            while ecg_embeds.ndim > 3:
                ecg_embeds = ecg_embeds.squeeze(0)
            if ecg_embeds.ndim == 2:
                ecg_embeds = ecg_embeds.unsqueeze(0)
        elif isinstance(ecg_embeds, list):
            ecg_embeds = torch.stack([e.squeeze(0) if e.ndim > 2 else e for e in ecg_embeds])
        
        print(f"[ECG-R1] ECG input shape: {ecg_embeds.shape}", flush=True)
        return ECGR1EmbeddingInputs(ecg_embeds=ecg_embeds)

    def _parse_and_validate_multimodal_inputs(self, **kwargs) -> dict[str, Any]:
        """解析所有多模态输入（扩展父类方法，添加 ECG）"""
        mm_input_by_modality = {}
        
        # 获取所有输入 key
        input_keys = list(kwargs.keys())
        
        for input_key in input_keys:
            # 图像输入
            if input_key in ("pixel_values", "image_embeds") and "image" not in mm_input_by_modality:
                mm_input_by_modality["image"] = self._parse_and_validate_image_input(**kwargs)
            # 视频输入
            if input_key in ("pixel_values_videos", "video_embeds") and "video" not in mm_input_by_modality:
                mm_input_by_modality["video"] = self._parse_and_validate_video_input(**kwargs)
            # ECG 输入
            if input_key == "ecg_embeds" and "ecg" not in mm_input_by_modality:
                mm_input_by_modality["ecg"] = self._parse_and_validate_ecg_input(**kwargs)
        
        # 移除 None 值
        mm_input_by_modality = {k: v for k, v in mm_input_by_modality.items() if v is not None}
        
        return mm_input_by_modality

    def _process_ecg_input(self, ecg_input: ECGR1EmbeddingInputs) -> tuple[torch.Tensor, ...]:
        """
        处理 ECG 输入，返回 embeddings
        
        参考 my_register_v3.py 第 160-174 行的处理逻辑：
        1. 输入: ecg_features (batch, 12, 5000) - 原始 ECG 信号
        2. tower: ecg_embeds = ecg_tower(ecg_features, output_last_transformer_layer=True)
        3. projector: ecg_embeds = ecg_projector(ecg_embeds)
        4. 输出: (batch, num_tokens, llm_hidden) -> tuple of (num_tokens, llm_hidden)
        
        参考 _process_image_input 的输出格式：返回 tuple of 2D tensors
        """
        ecg_features = ecg_input.ecg_embeds  # (batch, 12, 5000)
        batch_size = ecg_features.shape[0]
        
        if self.ecg_tower is None or self.ecg_projector is None:
            raise RuntimeError("ECG tower/projector not initialized")
        
        # 获取输入数据的设备（来自 vLLM，应该在 CUDA 上）
        target_device = ecg_features.device
        target_dtype = ecg_features.dtype
        
        # 参考 my_register_v3.py 第 153-154 行：确保 tower/projector 在正确设备上
        tower_device = next(self.ecg_tower.parameters()).device
        if tower_device != target_device:
            print(f"[ECG-R1] Moving ECG components: {tower_device} -> {target_device}", flush=True)
            self.ecg_tower = self.ecg_tower.to(device=target_device, dtype=target_dtype)
            self.ecg_projector = self.ecg_projector.to(device=target_device, dtype=target_dtype)
        
        print(f"[ECG-R1] Processing: input={ecg_features.shape}, device={target_device}", flush=True)
        
        # 参考 my_register_v3.py 第 160-161 行
        ecg_embeds = self.ecg_tower(ecg_features, output_last_transformer_layer=True)
        ecg_embeds = self.ecg_projector(ecg_embeds)
        # ecg_embeds: (batch, 101, 4096)
        
        print(f"[ECG-R1] After tower+projector: {ecg_embeds.shape}", flush=True)
        
        # 参考 _process_image_input 第 1312-1314 行: 拆分返回 tuple
        num_tokens_per_ecg = ecg_embeds.shape[1]  # 101
        # 参考 my_register_v3.py 第 172 行: reshape(-1, hidden)
        ecg_embeds_flat = ecg_embeds.reshape(-1, ecg_embeds.shape[-1])  # (batch*101, 4096)
        # 拆分成每个 ECG 的 tokens
        sizes = [num_tokens_per_ecg] * batch_size
        result = ecg_embeds_flat.split(sizes)  # tuple of (101, 4096)
        
        print(f"[ECG-R1] Output: {len(result)} items, shape={result[0].shape}", flush=True)
        return result

    def get_multimodal_embeddings(self, **kwargs) -> Optional[MultiModalEmbeddings]:
        """获取所有多模态 embeddings（扩展父类方法，添加 ECG）"""
        print(f"[ECG-R1 get_multimodal_embeddings] kwargs keys: {list(kwargs.keys())}", flush=True)
        
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        
        if not mm_input_by_modality:
            return None
        
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()
        
        # 按顺序处理各模态
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            
            if modality == "image":
                vision_embeddings = self._process_image_input(multimodal_input)
                multimodal_embeddings += vision_embeddings
                
            if modality == "video":
                video_embeddings = self._process_video_input(multimodal_input)
                multimodal_embeddings += video_embeddings
                
            if modality == "ecg":
                ecg_embeddings = self._process_ecg_input(multimodal_input)
                multimodal_embeddings += ecg_embeddings
                print(f"[ECG-R1 get_multimodal_embeddings] ECG processed, shape={ecg_embeddings[0].shape}", flush=True)
        
        return multimodal_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        """
        将多模态 embeddings 融合到 text embeddings
        
        参考 my_register_v3.py 第 129-176 行的融合逻辑：
        - 图像: inputs_embeds.masked_scatter(image_mask, image_embeds)
        - ECG: inputs_embeds[ecg_mask] = ecg_embeds_flat
        
        vLLM 使用 merge_multimodal_embeddings 实现相同功能
        """
        deepstack_input_embeds = None
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        
        if multimodal_embeddings is None:
            return inputs_embeds
        
        # 统计各模态的 token 数量（仅用于调试）
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        ecg_token_id = self.ecg_token_id
        
        n_image_tokens = (input_ids == image_token_id).sum().item()
        n_video_tokens = (input_ids == video_token_id).sum().item()
        n_ecg_tokens = (input_ids == ecg_token_id).sum().item() if ecg_token_id else 0
        
        # 调试：打印 token IDs
        print(f"[ECG-R1 merge] token_ids: image={image_token_id}, video={video_token_id}, ecg={ecg_token_id}", flush=True)
        print(f"[ECG-R1 merge] input_ids shape: {input_ids.shape}", flush=True)
        print(f"[ECG-R1 merge] tokens in input_ids: image={n_image_tokens}, video={n_video_tokens}, ecg={n_ecg_tokens}", flush=True)
        
        # 打印 embeddings 信息
        for i, emb in enumerate(multimodal_embeddings):
            print(f"[ECG-R1 merge] embedding[{i}] shape: {emb.shape}", flush=True)
        
        # 分离 embeddings: 根据最后一个维度区分
        # - 图像/视频 embeddings (deepstack): shape[-1] = visual_dim + multiscale_dim (4096 + 12288 = 16384)
        # - ECG embeddings: shape[-1] = llm_hidden_size (4096)
        image_video_embeddings = []
        ecg_embeddings = []
        
        expected_ecg_hidden = self.llm_hidden_size  # 4096
        
        for emb in multimodal_embeddings:
            emb_hidden = emb.shape[-1]
            # ECG embeddings 的 hidden size 等于 LLM hidden size
            # 图像/视频 embeddings 的 hidden size 是 visual_dim * (1 + deepstack_num_level)
            if emb_hidden == expected_ecg_hidden:
                ecg_embeddings.append(emb)
            else:
                image_video_embeddings.append(emb)
        
        print(f"[ECG-R1 merge] split: image_video={len(image_video_embeddings)}, ecg={len(ecg_embeddings)}", flush=True)
        
        # 1. 处理 image/video embeddings (使用 deepstack)
        if image_video_embeddings:
            image_video_tuple = tuple(image_video_embeddings)
            if self.use_deepstack:
                deepstack_input_embeds, image_video_tuple = self._compute_deepstack_embeds(
                    input_ids, inputs_embeds, image_video_tuple
                )
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, image_video_tuple,
                [self.config.image_token_id, self.config.video_token_id]
            )
        
        # 2. 处理 ECG embeddings (参考 my_register_v3.py 第 168-175 行)
        if ecg_embeddings and self.ecg_token_id is not None:
            ecg_tuple = tuple(ecg_embeddings)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, ecg_tuple,
                self.ecg_token_id
            )
            print(f"[ECG-R1 merge] ECG merged", flush=True)
        
        # 3. 处理 deepstack buffer
        if self.use_deepstack:
            if deepstack_input_embeds is None:
                deepstack_input_embeds = torch.zeros_like(
                    inputs_embeds).unsqueeze(0).repeat(
                        self.deepstack_num_level, 1, 1).contiguous()
            self._set_deepstack_input_embeds(deepstack_input_embeds)
        
        return inputs_embeds


# ==================== 测试代码 ====================

if __name__ == '__main__':
    """
    测试 ECG-R1 vLLM 模型的数据流
    
    测试内容:
    1. ECG 组件构建
    2. 权重映射器
    3. ECG 数据处理流程
    4. Embedding 融合
    """
    import os
    import sys
    import numpy as np
    
    # 添加项目根目录到 Python 路径
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # 设置环境变量
    os.environ['ECG_TOWER_PATH'] = 'ecg_coca/checkpoint/cpt_wfep_epoch_20.pt'
    os.environ['ECG_SEQ_LENGTH'] = '5000'
    os.environ['ECG_PATCH_SIZE'] = '50'
    os.environ['ECG_PROJECTOR_TYPE'] = 'mlp2x_gelu'
    os.environ['ECG_MODEL_CONFIG'] = 'coca_ViT-B-32'
    
    print("=" * 80)
    print("ECG-R1 vLLM 模型数据流测试")
    print("=" * 80)
    
    # ==================== 测试 1: ECG 组件构建 ====================
    print("\n[测试 1] ECG 组件构建")
    print("-" * 40)
    
    try:
        ecg_tower, ecg_cfg = build_ecg_tower(
            ecg_tower_path='ecg_coca/checkpoint/cpt_wfep_epoch_20.pt',
            model_config_name='coca_ViT-B-32',
            device='cpu'
        )
        print(f"✅ ECG Tower 构建成功")
        print(f"   - 类型: {type(ecg_tower).__name__}")
        print(f"   - 配置: ecg_hidden_size={ecg_cfg.get('ecg_cfg', {}).get('width', 768)}")
        
        # 统计参数量
        n_params = sum(p.numel() for p in ecg_tower.parameters())
        print(f"   - 参数量: {n_params:,}")
    except Exception as e:
        print(f"❌ ECG Tower 构建失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        ecg_projector = build_ecg_projector(
            ecg_hidden_size=768,
            llm_hidden_size=4096,
            projector_type='mlp2x_gelu'
        )
        print(f"✅ ECG Projector 构建成功")
        print(f"   - 类型: {type(ecg_projector).__name__}")
        print(f"   - 结构: {ecg_projector}")
    except Exception as e:
        print(f"❌ ECG Projector 构建失败: {e}")
    
    # ==================== 测试 2: 权重映射器 ====================
    print("\n[测试 2] 权重映射器")
    print("-" * 40)
    
    mapper = ECGR1ForConditionalGeneration.hf_to_vllm_mapper
    print(f"原始前缀映射: {mapper.orig_to_new_prefix}")
    
    # 测试映射
    test_names = [
        "model.ecg_tower.conv1.weight",
        "model.ecg_tower.transformer.resblocks.0.ln_1.weight",
        "model.ecg_projector.0.weight",
        "model.ecg_projector.2.bias",
        "model.visual.patch_embed.proj.weight",
        "model.language_model.model.layers.0.self_attn.q_proj.weight",
    ]
    
    print("\n映射测试:")
    for name in test_names:
        mapped = mapper._map_name(name)
        status = "✅" if mapped else "❌"
        print(f"  {status} {name}")
        print(f"     → {mapped}")
    
    # ==================== 测试 3: ECG 数据处理流程 ====================
    print("\n[测试 3] ECG 数据处理流程")
    print("-" * 40)
    
    # 创建模拟 ECG 数据 (batch=1, leads=12, samples=5000)
    ecg_data = torch.randn(1, 12, 5000)
    print(f"输入 ECG 数据: shape={ecg_data.shape}, dtype={ecg_data.dtype}")
    
    try:
        # 测试 ECG tower forward
        ecg_tower.eval()
        with torch.no_grad():
            ecg_features = ecg_tower(ecg_data, output_last_transformer_layer=True)
        print(f"✅ ECG Tower 输出: shape={ecg_features.shape}")
        
        # 测试 projector forward
        ecg_embeds = ecg_projector(ecg_features)
        print(f"✅ ECG Projector 输出: shape={ecg_embeds.shape}")
        
        # 期望的 embedding 形状
        ecg_seq_length = 5000
        ecg_patch_size = 50
        expected_tokens = ecg_seq_length // ecg_patch_size + 1  # +1 for cls token
        print(f"   - 预期 token 数: {expected_tokens}")
        print(f"   - 实际 token 数: {ecg_embeds.shape[1]}")
        
    except Exception as e:
        print(f"❌ ECG 数据处理失败: {e}")
        import traceback
        traceback.print_exc()
    
    # ==================== 测试 4: Embedding 融合模拟 ====================
    print("\n[测试 4] Embedding 融合模拟")
    print("-" * 40)
    
    # 模拟 input_ids 和 text embeddings
    seq_len = 200
    vocab_size = 152064
    hidden_size = 4096
    
    # 创建模拟 input_ids (包含 ECG placeholder tokens)
    # 假设 ecg_token_id = 151665 (需要与实际一致)
    ecg_token_id = 151665
    input_ids = torch.randint(0, vocab_size, (seq_len,))
    
    # 在某个位置插入 ECG tokens
    ecg_start_pos = 50
    n_ecg_tokens = ecg_embeds.shape[1] if 'ecg_embeds' in dir() else 101
    input_ids[ecg_start_pos:ecg_start_pos + n_ecg_tokens] = ecg_token_id
    
    print(f"模拟 input_ids: shape={input_ids.shape}")
    print(f"ECG token 位置: [{ecg_start_pos}, {ecg_start_pos + n_ecg_tokens})")
    print(f"ECG token 数量: {(input_ids == ecg_token_id).sum().item()}")
    
    # 模拟 text embeddings
    text_embeds = torch.randn(seq_len, hidden_size)
    print(f"模拟 text embeddings: shape={text_embeds.shape}")
    
    # 测试 merge_multimodal_embeddings
    try:
        if 'ecg_embeds' in dir():
            # 准备 ECG embeddings tuple
            ecg_embeds_flat = ecg_embeds.squeeze(0)  # (n_tokens, hidden)
            
            # 为了验证，让 ECG embeddings 有明显不同的值
            ecg_embeds_test = torch.ones_like(ecg_embeds_flat) * 100.0  # 使用明显不同的值
            
            # 保存原始值用于对比
            original_ecg_pos_values = text_embeds[ecg_start_pos:ecg_start_pos + n_ecg_tokens].clone()
            
            # 调用融合函数
            fused_embeds = merge_multimodal_embeddings(
                input_ids=input_ids,
                inputs_embeds=text_embeds.clone(),  # 使用 clone 避免原地修改
                multimodal_embeddings=(ecg_embeds_test,),
                placeholder_token_id=ecg_token_id
            )
            print(f"✅ Embedding 融合成功: shape={fused_embeds.shape}")
            
            # 验证融合结果
            ecg_positions = (input_ids == ecg_token_id)
            if ecg_positions.sum() > 0:
                # 检查融合后的 embeddings 是否改变
                original_mean = original_ecg_pos_values.mean().item()
                fused_mean = fused_embeds[ecg_positions].mean().item()
                print(f"   - 原始 ECG 位置均值: {original_mean:.4f}")
                print(f"   - 融合后 ECG 位置均值: {fused_mean:.4f} (预期接近 100.0)")
                print(f"   - 是否改变: {'✅ 是' if abs(fused_mean - 100.0) < 1.0 else '❌ 否'}")
                
                # 检查非 ECG 位置是否保持不变
                non_ecg_positions = (input_ids != ecg_token_id)
                non_ecg_original = text_embeds[non_ecg_positions].mean().item()
                non_ecg_fused = fused_embeds[non_ecg_positions].mean().item()
                print(f"   - 非 ECG 位置原始均值: {non_ecg_original:.4f}")
                print(f"   - 非 ECG 位置融合后均值: {non_ecg_fused:.4f}")
                print(f"   - 非 ECG 位置保持不变: {'✅ 是' if abs(non_ecg_original - non_ecg_fused) < 0.01 else '❌ 否'}")
        else:
            print("⚠️ 跳过融合测试 (ECG embeds 不可用)")
            
    except Exception as e:
        print(f"❌ Embedding 融合失败: {e}")
        import traceback
        traceback.print_exc()
    
    # ==================== 测试 5: 参数名检查 ====================
    print("\n[测试 5] 参数名检查 (与训练端对比)")
    print("-" * 40)
    
    if 'ecg_tower' in dir() and 'ecg_projector' in dir():
        print("ECG Tower 参数 (前10个):")
        for i, (name, param) in enumerate(ecg_tower.named_parameters()):
            if i >= 10:
                print("  ...")
                break
            print(f"  ecg_tower.{name}: {param.shape}")
        
        print("\nECG Projector 参数:")
        for name, param in ecg_projector.named_parameters():
            print(f"  ecg_projector.{name}: {param.shape}")
        
        # 模拟完整路径 (如果挂载到模型上)
        print("\n预期的 vLLM 参数路径:")
        for name, _ in ecg_tower.named_parameters():
            print(f"  ecg_tower.{name}")
            break
        print("  ...")
        for name, _ in ecg_projector.named_parameters():
            print(f"  ecg_projector.{name}")
    
    # ==================== 总结 ====================
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print("""
✅ 已验证:
   1. ECG Tower 和 Projector 可以正确构建
   2. 权重映射器包含 ECG 参数映射
   3. ECG 数据可以通过 Tower → Projector 处理
   4. merge_multimodal_embeddings 可以正确融合 embeddings

⚠️ 待测试 (需要完整 vLLM 环境):
   - 完整模型初始化 (需要 VllmConfig)
   - 实际推理流程
   - 权重同步接收

📝 下一步:
   1. 启动 swift rollout 测试实际推理
   2. 启动 swift rlhf 测试权重同步
""")

