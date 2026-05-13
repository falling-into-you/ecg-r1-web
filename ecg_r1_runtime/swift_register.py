import os

import sys
import types
import re
import random
import copy
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np

# ==================== 1. 基础依赖 ====================
from transformers import AutoConfig, AutoModel, AutoTokenizer
# 明确引入 Qwen3VL 相关类
from transformers import Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModelOutputWithPast

from swift.llm import (
    Model, ModelMeta, MultiModelKeys, Template, TemplateMeta,
    get_model_tokenizer, register_model, register_model_arch, register_template,
    get_template
)
# 直接引入 Qwen3VLTemplate
from swift.llm.template.template.qwen import Qwen3VLTemplate, QwenTemplateMeta
from swift.llm.template.template_inputs import StdTemplateInputs
from swift.llm.template.utils import Context, findall
from swift.utils import get_env_args, get_logger
logger = get_logger()

# 动态计算项目路径
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from ecg_r1_runtime.vllm_plugin import register as _register_vllm_plugin
    _register_vllm_plugin()
except Exception as e:
    print(f"[ECG-R1 Runtime] vLLM plugin registration skipped: {e}", flush=True)


# ==================== 2. ECG 组件构建工具 ====================

def build_ecg_tower(ecg_tower_path: str, model_config_name: str = 'coca_ViT-B-32', device: str = 'cpu'):
    """构建 ECG Tower"""
    from ecg_coca.training import get_ecg_encoder
    ecg_tower, ecg_processor, ecg_config = get_ecg_encoder(
        model_name=model_config_name,
        checkpoint_path=ecg_tower_path,
        device=device
    )
    logger.info(f'Loaded ECG tower from {ecg_tower_path}')
    return ecg_tower, ecg_config

def build_ecg_projector(ecg_hidden_size: int, llm_hidden_size: int, projector_type: str = 'mlp2x_gelu'):
    """构建 Projector"""
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

def load_ecg(ecg_path: str, ecg_seq_length: Optional[int] = 5000, root_ecg_dir: Optional[str] = None) -> torch.Tensor:
    """加载 ECG 数据 (WFDB)"""
    import wfdb
    if isinstance(ecg_path, torch.Tensor):
        return ecg_path
    
    path = ecg_path
    if root_ecg_dir and not os.path.isabs(path):
        path = os.path.join(root_ecg_dir, path)
    
    try:
        ecg_data = wfdb.rdsamp(path)[0]
    except Exception as e:
        raise ValueError(f"Failed to load ECG from {path}: {e}")
    
    ecg_data[np.isnan(ecg_data)] = 0
    ecg_data[np.isinf(ecg_data)] = 0
    # (L, C) -> (C, L)
    ecg_tensor = torch.from_numpy(np.transpose(ecg_data, (1, 0)).astype(np.float32))
    
    c, length = ecg_tensor.shape
    if ecg_seq_length is not None:
        if length < ecg_seq_length:
            new_tensor = torch.zeros((c, ecg_seq_length), dtype=ecg_tensor.dtype)
            new_tensor[:, 0:length] = ecg_tensor
            ecg_tensor = new_tensor
        elif length > ecg_seq_length:
            ecg_tensor = ecg_tensor[:, 0:ecg_seq_length]
    return ecg_tensor


# ==================== 3. 自定义 Backbone Forward 逻辑 ====================

def qwen3vl_backbone_forward_with_ecg(
    self, # self 指向 model.model (Backbone)
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Any] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    # 新增参数
    ecg_features: Optional[torch.FloatTensor] = None,
    **kwargs,
):
    """
    绑定到 Backbone 上的 Forward 方法。
    """
    from transformers.utils import is_torchdynamo_compiling
    
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)
        
        if input_ids is not None:
            input_ids = input_ids.to(inputs_embeds.device)

    # --- 1. 原生视觉 (Image) ---
    image_mask = None
    if pixel_values is not None:
        image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    # --- 2. 原生视觉 (Video) ---
    video_mask = None
    if pixel_values_videos is not None:
        video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    # --- 3. ECG 处理 (新增) ---
    if ecg_features is not None:
        if hasattr(self, 'ecg_tower') and hasattr(self, 'ecg_projector'):
            # 确保在同一设备
            ecg_tower_device = next(self.ecg_tower.parameters()).device
            ecg_features = ecg_features.to(ecg_tower_device, inputs_embeds.dtype)
            
            if not ecg_features.requires_grad:
                 ecg_features.requires_grad_(True)
            
            # Forward: Tower -> Projector
            ecg_embeds = self.ecg_tower(ecg_features, output_last_transformer_layer=True)
            ecg_embeds = self.ecg_projector(ecg_embeds)
            
            # 移回 embedding 设备
            ecg_embeds = ecg_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            
            # 融合逻辑
            ecg_token_id = getattr(self.config, 'ecg_token_id', None)
            if ecg_token_id is not None and input_ids is not None:
                ecg_mask = (input_ids == ecg_token_id)
                n_ecg_tokens = ecg_mask.sum()
                if n_ecg_tokens > 0:
                    ecg_embeds_flat = ecg_embeds.reshape(-1, ecg_embeds.shape[-1])
                    if ecg_embeds_flat.shape[0] >= n_ecg_tokens:
                        ecg_embeds_flat = ecg_embeds_flat[:n_ecg_tokens]
                        inputs_embeds[ecg_mask] = ecg_embeds_flat.to(inputs_embeds.dtype)
        else:
            logger.warning_once("ECG features provided but model has no ecg_tower attached.")

    # --- 4. DeepStack 准备 (原生 Qwen3VL 逻辑) ---
    visual_pos_masks = None
    deepstack_visual_embeds = None
    if image_mask is not None and video_mask is not None:
        image_mask = image_mask[..., 0]
        video_mask = video_mask[..., 0]
        visual_pos_masks = image_mask | video_mask
        deepstack_visual_embeds = []
        image_mask_joint = image_mask[visual_pos_masks]
        video_mask_joint = video_mask[visual_pos_masks]
        for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
            embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
            embed_joint[image_mask_joint, :] = img_embed
            embed_joint[video_mask_joint, :] = vid_embed
            deepstack_visual_embeds.append(embed_joint)
    elif image_mask is not None:
        image_mask = image_mask[..., 0]
        visual_pos_masks = image_mask
        deepstack_visual_embeds = deepstack_image_embeds
    elif video_mask is not None:
        video_mask = video_mask[..., 0]
        visual_pos_masks = video_mask
        deepstack_visual_embeds = deepstack_video_embeds

    # --- 5. RoPE 准备 (原生 Qwen3VL 逻辑) ---
    if position_ids is None:
        attention_mask_tensor = (
            attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
        )
        if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
            attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
            if attention_mask_tensor.dtype.is_floating_point:
                attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()

        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, attention_mask=attention_mask_tensor,
            )
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    # --- 6. 调用 LLM ---
    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
        **kwargs,
    )

    # --- 7. 返回 Qwen3VL 输出对象 ---
    return Qwen3VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )


# ==================== 4. ECGR1ForConditionalGeneration 类 ====================

class ECGR1ForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """
    ECG-R1 模型类，继承自 Qwen3VLForConditionalGeneration。
    内部自动挂载 ECG 组件，并劫持 backbone 的 forward 逻辑。
    """
    def __init__(self, config):
        super().__init__(config)
        
        # 1. 初始化 ECG 组件
        self._init_ecg_components(config)
        
        # 2. 绑定自定义 forward 到 backbone (self.model)
        if hasattr(self, 'model'):
            self.model.forward = types.MethodType(qwen3vl_backbone_forward_with_ecg, self.model)
            logger.info('✅ ECGR1: Bound custom forward method to backbone model.')
        else:
            logger.error('❌ ECGR1: self.model not found, initialization failed.')

    def _init_ecg_components(self, config):
        ecg_tower_path = getattr(config, 'ecg_tower_path', None) or get_env_args('ECG_TOWER_PATH', str, None)
        ecg_projector_type = getattr(config, 'ecg_projector_type', None) or get_env_args('ECG_PROJECTOR_TYPE', str, 'mlp2x_gelu')
        ecg_model_config = getattr(config, 'ecg_model_config', None) or get_env_args('ECG_MODEL_CONFIG', str, 'coca_ViT-B-32')

        llm_hidden_size = getattr(config, 'hidden_size', None)
        if llm_hidden_size is None and hasattr(config, 'text_config'):
             llm_hidden_size = getattr(config.text_config, 'hidden_size', None)
        
        if ecg_tower_path and llm_hidden_size:
            # 避免重复加载
            if hasattr(self.model, 'ecg_tower'):
                return

            logger.info(f'Initializing ECG components from {ecg_tower_path}...')
            try:
                ecg_tower, ecg_cfg = build_ecg_tower(ecg_tower_path, ecg_model_config, device='cpu')
                ecg_hidden_size = ecg_cfg.get('ecg_cfg', {}).get('width', 768)
                ecg_projector = build_ecg_projector(ecg_hidden_size, llm_hidden_size, ecg_projector_type)
                
                # 挂载
                self.model.ecg_tower = ecg_tower
                self.model.ecg_projector = ecg_projector
                
                # 保存 Config
                config.ecg_tower_path = ecg_tower_path
                config.ecg_projector_type = ecg_projector_type
                config.ecg_model_config = ecg_model_config
                config.ecg_hidden_size = ecg_hidden_size
                
                logger.info('✅ ECG components attached successfully.')
            except Exception as e:
                logger.error(f'❌ Failed to initialize ECG components: {e}')
                raise e

    def forward(self, ecg_features: Optional[torch.FloatTensor] = None, **kwargs):
        """
        外层 forward。显式接收 ecg_features 并透传。
        """
        return super().forward(ecg_features=ecg_features, **kwargs)


# ==================== 5. ECGR1Template 定义 ====================

class ECGR1Template(Qwen3VLTemplate):
    """
    ECG-R1 模板，继承自 Qwen3VLTemplate。
    """
    version = 'v3'
    ecg_placeholder = '<|ecg_pad|>'
    ecg_start_token = '<|ecg_start|>'
    ecg_end_token = '<|ecg_end|>'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ecg_seq_length = get_env_args('ECG_SEQ_LENGTH', int, 5000)
        self.ecg_patch_size = get_env_args('ECG_PATCH_SIZE', int, 50)
        self.ecg_num_patches = self.ecg_seq_length // self.ecg_patch_size
        self.root_ecg_dir = get_env_args('ROOT_ECG_DIR', str, None)
        self.interleave_prob = get_env_args('INTERLEAVE_PROB', float, 0.1)
        self.modality_dropout_prob = get_env_args('MODALITY_DROPOUT_PROB', float, 0.5)
        try:
            seed = torch.initial_seed()
        except Exception:
            seed = 42
        self._rng = random.Random(seed)
        # ECG Token ID 将在 init_processor 中设置，因为此时 processor 可能还未初始化

    def init_processor(self, processor) -> None:
        """重写 init_processor，在 processor 设置后注册 ECG Token ID"""
        super().init_processor(processor)
        # 现在 processor 已经设置，可以安全地访问它
        if hasattr(self, 'processor') and self.processor is not None:
            tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else self.processor
            self.ecg_token_id = tokenizer.convert_tokens_to_ids(self.ecg_placeholder)
            self.ecg_start_token_id = tokenizer.convert_tokens_to_ids(self.ecg_start_token)
            self.ecg_end_token_id = tokenizer.convert_tokens_to_ids(self.ecg_end_token)
            
            if self.ecg_token_id not in self.placeholder_tokens:
                self.placeholder_tokens.append(self.ecg_token_id)

    def replace_ecg(self, ecg_data: Any, index: int, inputs: StdTemplateInputs) -> List[Context]:
        """加载数据并返回占位符"""
        ecgs = inputs.objects.get('ecg', [])
        if index < len(ecgs):
            ecg = ecgs[index]
            if isinstance(ecg, str):
                ecgs[index] = load_ecg(ecg, self.ecg_seq_length, self.root_ecg_dir)
        return [self.ecg_start_token, self.ecg_placeholder, self.ecg_end_token]

    # ===== Interleave & Modality Dropout helpers =====
    def _remove_ecg_tag(self, text: str) -> str:
        text = re.sub(r'\s*<ecg>\s*', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()

    def _remove_image_tag(self, text: str) -> str:
        text = re.sub(r'\s*<image>\s*', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()

    def _swap_ecg_image(self, text: str) -> str:
        # Try ecg->image, if none swapped try image->ecg
        new_text, n = re.subn(r'<ecg>(\s*)<image>', r'<image>\1<ecg>', text)
        if n == 0:
            new_text, _ = re.subn(r'<image>(\s*)<ecg>', r'<ecg>\1<image>', text)
        return new_text

    def _restore_one_modality(self, inputs: StdTemplateInputs, orig_messages, orig_ecg, orig_images, prefer: str = 'image'):
        if prefer == 'image' and orig_images:
            inputs.images = copy.deepcopy(orig_images)
        if prefer == 'ecg' and orig_ecg:
            inputs.objects['ecg'] = copy.deepcopy(orig_ecg)
        # 如果仍为空，回退到原始
        if not inputs.images and orig_images:
            inputs.images = copy.deepcopy(orig_images)
        if (not inputs.objects.get('ecg')) and orig_ecg:
            inputs.objects['ecg'] = copy.deepcopy(orig_ecg)
        inputs.messages = copy.deepcopy(orig_messages)
        return inputs

    def _maybe_interleave_and_dropout(self, inputs: StdTemplateInputs) -> StdTemplateInputs:
        if self.mode != 'train':
            return inputs

        # 备份原始内容以便回退
        orig_messages = copy.deepcopy(inputs.messages)
        orig_ecg = copy.deepcopy(inputs.objects.get('ecg', []))
        orig_images = copy.deepcopy(getattr(inputs, 'images', []))

        rng = self._rng
        has_ecg = bool(orig_ecg)
        has_img = bool(orig_images)


        # 单概率模态丢弃：在可用模态中随机选一侧
        if rng.random() < self.modality_dropout_prob and (has_ecg or has_img):
            candidates = []
            if has_ecg:
                candidates.append('ecg')
            if has_img:
                candidates.append('image')
            if candidates:
                choice = rng.choice(candidates)
                if choice == 'ecg':
                    inputs.objects['ecg'] = []
                    inputs.messages = [
                        {**m, 'content': self._remove_ecg_tag(m['content'])} if m.get('role') == 'user' else m
                        for m in inputs.messages
                    ]
                elif choice == 'image':
                    inputs.images = []
                    inputs.messages = [
                        {**m, 'content': self._remove_image_tag(m['content'])} if m.get('role') == 'user' else m
                        for m in inputs.messages
                    ]

        # 顺序随机化
        if rng.random() < self.interleave_prob:
            inputs.messages = [
                {**m, 'content': self._swap_ecg_image(m['content'])} if m.get('role') == 'user' else m
                for m in inputs.messages
            ]

        # 守护：避免两模态都被去除
        if not inputs.objects.get('ecg') and not getattr(inputs, 'images', []):
            inputs = self._restore_one_modality(inputs, orig_messages, orig_ecg, orig_images, prefer='image')

        return inputs

    def _pre_tokenize(self, context_list: List[Context], loss_scale_list: List[float], inputs: StdTemplateInputs):
        """
        1. 处理 <ecg> 标签：拆分并替换为 token。
        2. 调用 super()._pre_tokenize()，由父类 Qwen3VLTemplate 处理剩余的 image/video。
        """
        new_ctx, new_loss = [], []
        inputs.ecg_idx = 0 # 确保索引从0开始

        for ctx, loss in zip(context_list, loss_scale_list):
            if isinstance(ctx, str) and '<ecg>' in ctx:
                parts = re.split(r'(<ecg>)', ctx)
                for part in parts:
                    if part == '<ecg>':
                        # 替换为 ECG Tokens，loss 置为 0
                        c_list = self.replace_ecg(None, inputs.ecg_idx, inputs)
                        inputs.ecg_idx += 1
                        new_ctx.extend(c_list)
                        new_loss.extend([0.0] * len(c_list))
                    elif part: # 非空字符串
                        new_ctx.append(part)
                        new_loss.append(loss)
            else:
                new_ctx.append(ctx)
                new_loss.append(loss)
        
        # 将处理完 ECG 的列表传给父类，父类会处理剩下的 <image>/<video>
        return super()._pre_tokenize(new_ctx, new_loss, inputs)

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        """
        1. 调用 super()._encode() 生成 input_ids 和 visual tensor。
        2. 补充 ECG 特有的 Tensor 堆叠和 ID 扩展逻辑。
        3. 设置 mm_processor_kwargs 以便 vLLM 使用正确的图像参数。
        """
        inputs = self._maybe_interleave_and_dropout(inputs)
        encoded = super()._encode(inputs)
        
        # 调用 ECG 后处理
        return self._postprocess_ecg(encoded, inputs)
    
    def _encode_truncated(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        """
        重写 _encode_truncated 以确保在 vLLM 模式下也处理 ECG 数据。
        
        ⚠️ 关键：父类在 vLLM 模式下会跳过子类的 _encode()，直接调用 Template._encode()。
        我们需要在这里添加 ECG 后处理逻辑。
        """
        # 调用父类的 _encode_truncated
        inputs = self._maybe_interleave_and_dropout(inputs)
        encoded = super()._encode_truncated(inputs)
        
        # 如果是 vLLM 模式，父类跳过了我们的 _encode，需要手动处理 ECG
        if self.mode in {'vllm', 'lmdeploy', 'sglang'}:
            encoded = self._postprocess_ecg(encoded, inputs)
        
        return encoded
    
    def _postprocess_ecg(self, encoded: Dict[str, Any], inputs: StdTemplateInputs) -> Dict[str, Any]:
        """
        ECG 后处理：扩展 token 和加载数据。
        抽取为独立方法，供 _encode 和 _encode_truncated 调用。
        """
        # 设置 mm_processor_kwargs (让 vLLM 使用正确的图像参数)
        # vLLM 不会调用 patch_qwen_vl_utils，需要显式传递这些参数
        # 注意：需要同时设置 inputs 和 encoded，因为在 vLLM 模式下父类可能已经处理过
        factor = 32  # patch_size(16) × merge_size(2) for Qwen3VL
        max_tokens = int(os.environ.get('IMAGE_MAX_TOKEN_NUM', '768'))
        min_tokens = int(os.environ.get('IMAGE_MIN_TOKEN_NUM', '4'))
        mm_processor_kwargs = {
            'min_pixels': min_tokens * (factor ** 2),  # 4 × 32² = 4,096
            'max_pixels': max_tokens * (factor ** 2),  # 768 × 32² = 786,432
        }
        inputs.mm_processor_kwargs = mm_processor_kwargs
        encoded['mm_processor_kwargs'] = mm_processor_kwargs  # 确保 vLLM 模式下也生效
        
        input_ids = encoded['input_ids']
        
        ecgs = inputs.objects.get('ecg', [])
        if ecgs:
            # 扩展 Token ID (1个 placeholder -> N+1 个真实 token)
            idx_list = findall(input_ids, self.ecg_token_id)
            if idx_list:
                tokens_per_ecg = self.ecg_num_patches + 1 # +1 是 cls token
                def _get_tokens(i): return [self.ecg_token_id] * tokens_per_ecg
                
                input_ids, encoded['labels'], encoded['loss_scale'] = self._extend_tokens(
                    input_ids, encoded['labels'], encoded.get('loss_scale'), idx_list, _get_tokens
                )
            
            # 堆叠 Tensor
            tensor_list = []
            for item in ecgs:
                if isinstance(item, str): 
                    item = load_ecg(item, self.ecg_seq_length, self.root_ecg_dir)
                tensor_list.append(item)
            
            if tensor_list:
                encoded['ecg_features'] = torch.stack(tensor_list)
        
        encoded['input_ids'] = input_ids
        return encoded
    
    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate 时拼接 ECG Features"""
        res = super()._data_collator_mm_data(batch)
        ecg_features = [b['ecg_features'] for b in batch if b.get('ecg_features') is not None]
        if ecg_features:
            res['ecg_features'] = torch.cat(ecg_features, dim=0)
        return res
    
    def normalize_bbox(self, inputs: StdTemplateInputs):
        """
        [Fix] 重写以防止 KeyError: 'bbox'
        父类逻辑假设 inputs.objects 非空就一定包含 bbox，
        但我们这里可能只有 ecg 数据。
        """
        if inputs.objects and 'bbox' in inputs.objects:
            return super().normalize_bbox(inputs)
        
        # 如果没有 bbox (例如只有 ecg)，什么都不做，直接返回
        return
    
register_template(
    QwenTemplateMeta(
        'ecg_r1',
        template_cls=ECGR1Template,
        default_system='You are a helpful, harmless clinical ECG assistant. Provide concise, evidence-based interpretations.',
    ))


# ==================== 6. 注册与测试入口 ====================

register_model_arch(
    MultiModelKeys(
        'ecg_r1',
        language_model='model.language_model',
        vision_tower=['model.visual', 'model.ecg_tower'],
        aligner=['model.visual.merger', 'model.visual.deepstack_merger_list', 'model.ecg_projector'],
    )
)

def get_model_tokenizer_ecg_r1(model_dir, model_info, model_kwargs, load_model=True, **kwargs):
    kwargs['automodel_class'] = ECGR1ForConditionalGeneration
    kwargs['_check_qwen_vl_utils'] = False 
    
    from swift.llm.model.model.qwen import get_model_tokenizer_qwen2_vl
    model, processor = get_model_tokenizer_qwen2_vl(model_dir, model_info, model_kwargs, load_model, **kwargs)
    
    # 添加 Special Tokens
    if processor is not None:
        tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
        ecg_tokens = ['<|ecg_pad|>', '<|ecg_start|>', '<|ecg_end|>']
        tokens_to_add = [t for t in ecg_tokens if t not in tokenizer.get_vocab()]
        
        if tokens_to_add:
            num = tokenizer.add_special_tokens({'additional_special_tokens': tokens_to_add})
            if model is not None and num > 0:
                model.resize_token_embeddings(len(tokenizer))
                model.config.ecg_token_id = tokenizer.convert_tokens_to_ids('<|ecg_pad|>')

    # 设置冻结状态
    if model and load_model and hasattr(model.model, 'ecg_tower'):
        # 1. 获取环境变量控制 (默认都训练)
        freeze_tower = get_env_args('FREEZE_ECG_TOWER', bool, False)
        freeze_projector = get_env_args('FREEZE_ECG_PROJECTOR', bool, False)
        
        # 2. 定义统一的处理函数
        def _set_module_state(module, is_frozen, name):
            if is_frozen:
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
                logger.info(f"{name}: Frozen (eval mode, requires_grad=False)")
            else:
                module.train()
                for param in module.parameters():
                    param.requires_grad = True
                logger.info(f"{name}: Trainable (train mode, requires_grad=True)")

        # 3. 应用状态
        _set_module_state(model.model.ecg_tower, freeze_tower, "ECG Tower")
        _set_module_state(model.model.ecg_projector, freeze_projector, "ECG Projector")

    return model, processor

register_model(ModelMeta(
    'ecg_r1', [], 'ecg_r1', get_model_tokenizer_ecg_r1, 
    is_multimodal=True, model_arch='ecg_r1', 
    architectures=['Qwen3VLForConditionalGeneration', 'ECGR1ForConditionalGeneration'], 
    tags=['vision', 'ecg']
))
