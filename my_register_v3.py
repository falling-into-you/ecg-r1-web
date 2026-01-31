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
        # Prioritize environment variable to allow overriding config (which might contain relative paths)
        ecg_tower_path = get_env_args('ECG_TOWER_PATH', str, None) or getattr(config, 'ecg_tower_path', None)
        ecg_projector_type = get_env_args('ECG_PROJECTOR_TYPE', str, None) or getattr(config, 'ecg_projector_type', 'mlp2x_gelu')
        ecg_model_config = get_env_args('ECG_MODEL_CONFIG', str, None) or getattr(config, 'ecg_model_config', 'coca_ViT-B-32')

        llm_hidden_size = getattr(config, 'hidden_size', None)
        if llm_hidden_size is None and hasattr(config, 'text_config'):
             llm_hidden_size = getattr(config.text_config, 'hidden_size', None)
        
        if ecg_tower_path and llm_hidden_size:
            # 避免重复加载
            if hasattr(self.model, 'ecg_tower'):
                return

            logger.info(f'Initializing ECG components from {ecg_tower_path}...')
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                ecg_tower, ecg_cfg = build_ecg_tower(ecg_tower_path, ecg_model_config, device=device)
                ecg_hidden_size = ecg_cfg.get('ecg_cfg', {}).get('width', 768)
                ecg_projector = build_ecg_projector(ecg_hidden_size, llm_hidden_size, ecg_projector_type)
                
                # Handle meta device initialization (accelerate)
                if any(p.device.type == 'meta' for p in ecg_projector.parameters()):
                    ecg_projector.to_empty(device=device)
                    # Re-initialize parameters since to_empty() leaves them uninitialized
                    def _init_weights(m):
                        if isinstance(m, nn.Linear):
                            nn.init.xavier_uniform_(m.weight)
                            if m.bias is not None:
                                nn.init.zeros_(m.bias)
                    ecg_projector.apply(_init_weights)
                    logger.info(f"Initialized ECG projector weights on {device} (was meta)")
                else:
                    ecg_projector = ecg_projector.to(device)
                
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
        default_system='You are a helpful assistant.',
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

if __name__ == '__main__':
    # 限制只使用第一张 GPU，避免多卡设备不一致问题
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    os.environ['ECG_SEQ_LENGTH'] = '5000'
    os.environ['ECG_PATCH_SIZE'] = '50'
    os.environ['ROOT_ECG_DIR'] = '/data/jinjiarui/datasets/ECG_R1_Dataset/ecg_timeseries'
    os.environ['ROOT_IMAGE_DIR'] = "/data/jinjiarui/datasets/ECG_R1_Dataset/ecg_images"
    os.environ['IMAGE_MAX_TOKEN_NUM'] = '768'
    os.environ['ECG_TOWER_PATH'] = 'ecg_coca/checkpoint/cpt_wfep_epoch_20.pt'
    os.environ['ECG_PROJECTOR_TYPE'] = 'mlp2x_gelu'
    os.environ['ECG_MODEL_CONFIG'] = 'coca_ViT-B-32'
    # 关闭 interleave/dropout，避免测试时缺模态
    os.environ['INTERLEAVE_PROB'] = '0'
    os.environ['MODALITY_DROPOUT_PROB'] = '0'
    
    # 设置 ECG 训练参数（可选，取消注释以启用）
    os.environ['FREEZE_ECG_TOWER'] = 'True'  # 训练 ECG tower
    os.environ['FREEZE_ECG_PROJECTOR'] = 'False'  # 训练 ECG projector

    # 测试与debug
    model, processor = get_model_tokenizer('Qwen/Qwen3-VL-8B-Instruct', model_type='ecg_r1')
    

    # 检查 ECG 组件是否加载
    has_ecg_tower = hasattr(model.model, 'ecg_tower')
    has_ecg_projector = hasattr(model.model, 'ecg_projector')
    print(f'\n🔍 ECG Components Status:')
    print(f'   ECG Tower loaded: {has_ecg_tower}')
    print(f'   ECG Projector loaded: {has_ecg_projector}')
    if not has_ecg_tower:
        print(f'   ⚠️  Tip: Set ECG_TOWER_PATH environment variable to load ECG tower')
    print()
    
    template = get_template('ecg_r1', processor)
    # 确保测试时不做随机丢弃/互换
    template.interleave_prob = 0.0
    template.modality_dropout_prob = 0.0
    template._rng = random.Random(42)
    data = {
        'messages': [
            {'role': 'user', 'content': '<ecg><image>\nTime for a multiple-choice challenge! Share your thought process, then lock in your final answer.\nWhat can be inferred about the cardiac axis on this ECG?\nA. The axis is normal\nB. The axis is deviated to the right\nC. The axis is indeterminate\nD. The axis is deviated to the left'},
            {'role': 'assistant', 'content': 'The ECG shows no clear indication of a specific axis deviation, and the QRS morphology does not suggest a clear right or left axis deviation. This image indicates that the cardiac axis is indeterminate, meaning that it cannot be determined based on the ECG findings.\n\nTherefore, we choose C. The axis is indeterminate'},
        ],
        'images': ['mimic/p1127/p11273115/s44111511/44111511-0.png'],
        'objects': {'ecg': ['mimic-iv/files/p1127/p11273115/s44111511/44111511']},
    }
    template.set_mode('train')
    encoded = template.encode(data)
    
    # 检查 ECG token
    ecg_token_ids = template._tokenize(template.ecg_placeholder)
    image_token_id = template.image_token_id
    print(f'\n=== Token Info ===')
    print(f'ECG placeholder: {template.ecg_placeholder}')
    print(f'ECG token IDs: {ecg_token_ids}')
    print(f'Image token ID: {image_token_id}')
    print(f'placeholder_tokens: {template.placeholder_tokens}')
    
    # 统计 input_ids 中的 token
    input_ids_list = encoded['input_ids']
    labels_list = encoded['labels']
    if isinstance(ecg_token_ids, list) and len(ecg_token_ids) > 0:
        ecg_token_id = ecg_token_ids[0]
        ecg_count = input_ids_list.count(ecg_token_id)
        print(f'ECG token {ecg_token_id} appears {ecg_count} times in input_ids')
    image_count = input_ids_list.count(image_token_id)
    print(f'Image token {image_token_id} appears {image_count} times in input_ids')
    
    print('\n=== Decoded ===')
    print('input_ids: ' + template.safe_decode(encoded['input_ids']))
    print('labels: ' + template.safe_decode(encoded['labels']))
    print('keys: ' + str(encoded.keys()))
    
    # 打印详细信息
    print(f'\n=== Detailed Info ===')
    print(f'input_ids length: {len(encoded["input_ids"])}')
    print(f'labels length: {len(encoded["labels"])}')
    if 'ecg_features' in encoded:
        print(f'ecg_features shape: {encoded["ecg_features"].shape}')
    if 'pixel_values' in encoded:
        print(f'pixel_values shape: {encoded["pixel_values"].shape}')
    if 'image_grid_thw' in encoded:
        print(f'image_grid_thw: {encoded["image_grid_thw"]}')
    
    # ========== 详细的 Label Mask 验证 ==========
    print(f'\n{"="*80}')
    print('=== Label Mask Validation ===')
    print(f'{"="*80}')
    
    # 1. 统计 labels 中的 -100（不计算loss的位置）
    num_ignore = sum(1 for label in labels_list if label == -100)
    num_train = len(labels_list) - num_ignore
    print(f'\n1. Label Statistics:')
    print(f'   Total tokens: {len(labels_list)}')
    print(f'   Ignored tokens (label=-100): {num_ignore} ({num_ignore/len(labels_list)*100:.1f}%)')
    print(f'   Training tokens (label!=-100): {num_train} ({num_train/len(labels_list)*100:.1f}%)')
    
    # 2. 验证特殊 token 的 label 是否为 -100
    print(f'\n2. Special Token Label Check:')
    special_tokens = {
        'ECG pad': template.ecg_token_id,
        'ECG start': template.ecg_start_token_id,
        'ECG end': template.ecg_end_token_id,
        'Image': template.image_token_id,
        'Vision start': template.processor.tokenizer.convert_tokens_to_ids('<|vision_start|>'),
        'Vision end': template.processor.tokenizer.convert_tokens_to_ids('<|vision_end|>'),
    }
    
    for name, token_id in special_tokens.items():
        positions = [i for i, tok in enumerate(input_ids_list) if tok == token_id]
        if positions:
            label_values = [labels_list[i] for i in positions[:5]]  # 只显示前5个
            all_masked = all(labels_list[i] == -100 for i in positions)
            status = '✓ All masked' if all_masked else '✗ Some not masked'
            print(f'   {name:15} (ID={token_id:6}): {len(positions):4} occurrences, {status}')
            if len(positions) <= 5:
                print(f'      Labels at positions {positions}: {label_values}')
    
    # 3. 找到 assistant 的回复部分
    print(f'\n3. Assistant Response Check:')
    tokenizer = template.processor.tokenizer
    im_start_id = tokenizer.convert_tokens_to_ids('<|im_start|>')
    im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    
    # 找到最后一个 <|im_start|>assistant
    assistant_token = tokenizer.encode('assistant', add_special_tokens=False)[0]
    assistant_start = None
    for i in range(len(input_ids_list) - 1):
        if input_ids_list[i] == im_start_id and input_ids_list[i+1] == assistant_token:
            assistant_start = i
    
    if assistant_start is not None:
        # 找到对应的 <|im_end|>
        assistant_end = None
        for i in range(assistant_start + 1, len(input_ids_list)):
            if input_ids_list[i] == im_end_id:
                assistant_end = i
                break
        
        if assistant_end is not None:
            assistant_content = input_ids_list[assistant_start:assistant_end+1]
            assistant_labels = labels_list[assistant_start:assistant_end+1]
            
            # 统计 assistant 部分的 label
            assistant_ignore = sum(1 for label in assistant_labels if label == -100)
            assistant_train = len(assistant_labels) - assistant_ignore
            
            print(f'   Assistant tokens range: [{assistant_start}, {assistant_end}] (length={len(assistant_content)})')
            print(f'   Assistant ignored tokens: {assistant_ignore}')
            print(f'   Assistant training tokens: {assistant_train}')
            
            # 显示前几个 token 的 label
            print(f'\n   First 10 tokens in assistant response:')
            for i in range(min(10, len(assistant_content))):
                idx = assistant_start + i
                token_str = tokenizer.decode([input_ids_list[idx]])
                label_str = 'IGNORE' if labels_list[idx] == -100 else str(labels_list[idx])
                print(f'      [{idx:4}] Token: {token_str:20} | Label: {label_str}')
    
    # 4. 验证 input_ids 和 labels 的对齐
    print(f'\n4. Input-Label Alignment Check:')
    misaligned = []
    for i in range(len(input_ids_list)):
        if labels_list[i] != -100 and labels_list[i] != input_ids_list[i]:
            misaligned.append((i, input_ids_list[i], labels_list[i]))
    
    if misaligned:
        print(f'   ✗ Found {len(misaligned)} misaligned positions:')
        for pos, input_id, label in misaligned[:5]:
            print(f'      Position {pos}: input_id={input_id}, label={label}')
    else:
        print(f'   ✓ All labels are either -100 or equal to input_ids (correct!)')
    
    # 5. 检查 loss_scale（如果有）
    if 'loss_scale' in encoded and encoded['loss_scale'] is not None:
        print(f'\n5. Loss Scale Check:')
        loss_scale = encoded['loss_scale']
        print(f'   Loss scale length: {len(loss_scale)}')
        unique_scales = set(loss_scale)
        print(f'   Unique loss scale values: {sorted(unique_scales)}')
        for scale in sorted(unique_scales):
            count = sum(1 for s in loss_scale if s == scale)
            print(f'      Scale {scale}: {count} tokens ({count/len(loss_scale)*100:.1f}%)')
    
    print(f'\n{"="*80}')
    print('=== Validation Complete ===')
    print(f'{"="*80}\n')
    
    # ========== 测试 Forward Pass ==========
    if hasattr(model.model, 'ecg_tower'):
        print(f'{"="*80}')
        print('=== Forward Pass Test ===')
        print(f'{"="*80}')
        
        try:
            import torch
            
            # --- 修复 1：动态获取实际设备 ---
            # 找到 ECG Tower 所在的实际设备 (即权重所在的设备，这通常是 Accelerate 放置的设备)
            device = next(model.model.ecg_tower.parameters()).device
            print(f"🎯 Target Device determined from ECG Tower: {device}")
            
            # 确保 model 的所有子模块都位于该设备（虽然 Accelerate 会处理，但手动统一更保险）
            if str(device).startswith('cuda'):
                model.to(device)
            
            # --- 修复 2：统一输入数据的设备 ---
            inputs = {
                'input_ids': torch.tensor([encoded['input_ids']]).to(device),
                'labels': torch.tensor([encoded['labels']]).to(device),
            }
            if 'pixel_values' in encoded:
                inputs['pixel_values'] = encoded['pixel_values'].unsqueeze(0).to(device)
            if 'image_grid_thw' in encoded:
                inputs['image_grid_thw'] = encoded['image_grid_thw'].to(device)
            if 'ecg_features' in encoded:
                inputs['ecg_features'] = encoded['ecg_features'].to(device)
            
            # 确保 attention mask 存在且在目标设备上
            seq_len = inputs['input_ids'].shape[1]
            inputs['attention_mask'] = torch.ones(1, seq_len).to(device)
            
            print(f'\n1. Input Shapes:')
            for key, val in inputs.items():
                if isinstance(val, torch.Tensor):
                    print(f'   {key:20}: {list(val.shape)}')
            
            # Forward pass
            print(f'\n2. Running forward pass...')
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)
            
            print(f'   ✓ Forward pass successful!')
            
            # 检查输出
            print(f'\n3. Output Information:')
            print(f'   Loss: {outputs.loss.item():.4f}')
            print(f'   Logits shape: {list(outputs.logits.shape)}')
            
            # 验证 ECG embeddings
            # 注意：由于在 forward 中才进行融合，这里无法直接看到 embedding 融合后的结果
            # 但我们可以检查 logits
            print(f'\n4. ECG Processing Verification:')
            
            # 计算训练 token 的平均 loss
            trainable_positions = [i for i, label in enumerate(encoded['labels']) if label != -100]
            if trainable_positions:
                print(f'\n5. Training Token Analysis:')
                print(f'   Trainable positions: {len(trainable_positions)}')
                
                # 计算 per-token loss
                shift_logits = outputs.logits[0, :-1, :]
                shift_labels = inputs['labels'][0, 1:]
                
                # 只计算非 -100 的 token
                mask = shift_labels != -100
                if mask.sum() > 0:
                    from torch.nn import functional as F
                    token_losses = F.cross_entropy(
                        shift_logits[mask], 
                        shift_labels[mask], 
                        reduction='none'
                    )
                    print(f'   Per-token loss stats:')
                    print(f'      Mean: {token_losses.mean().item():.4f}')
                    print(f'      Min:  {token_losses.min().item():.4f}')
                    print(f'      Max:  {token_losses.max().item():.4f}')
            
            print(f'\n   ✅ All forward pass tests passed!')
            
        except Exception as e:
            print(f'\n   ❌ Forward pass failed with error:')
            print(f'   Error type: {type(e).__name__}')
            print(f'   Error message: {str(e)}')
            import traceback
            print(f'\n   Traceback:')
            traceback.print_exc()
        
        print(f'\n{"="*80}\n')
    else:
        print(f'\n⚠️  Skipping forward pass test (ECG tower not loaded)\n')

    # ========== 单元测试：Interleave & Dropout 辅助函数 ==========
    print(f'{"="*80}')
    print('=== Interleave & Dropout Helper Tests ===')
    print(f'{"="*80}')

    # 基础样本：简单占位符示例，避免干扰主数据
    sample_inputs = StdTemplateInputs(
        messages=[{'role': 'user', 'content': '<ecg> <image>'}],
        images=['img'],
        objects={'ecg': ['ecg']}
    )

    # 1) 互换占位符
    swapped = template._swap_ecg_image(sample_inputs.messages[0]['content'])
    print(f'swap_ecg_image: "{sample_inputs.messages[0]["content"]}" -> "{swapped}"')

    # 2) 删除 ECG tag
    removed_ecg = template._remove_ecg_tag('<ecg> hello <image>')
    print(f'remove_ecg_tag: "<ecg> hello <image>" -> "{removed_ecg}"')

    # 3) 删除 image tag
    removed_img = template._remove_image_tag('<ecg> hello <image>')
    print(f'remove_image_tag: "<ecg> hello <image>" -> "{removed_img}"')

    # 4) 模态丢弃守护：同时清空应恢复一侧
    tmp_inputs = StdTemplateInputs(
        messages=[{'role': 'user', 'content': '<ecg> <image>'}],
        objects={'ecg': ['ecg']},
        images=['img']
    )
    tmp_inputs.objects['ecg'] = []
    tmp_inputs.images = []
    restored = template._restore_one_modality(
        tmp_inputs,
        orig_messages=[{'role': 'user', 'content': '<ecg> <image>'}],
        orig_ecg=['ecg'],
        orig_images=['img'],
        prefer='image'
    )
    assert restored.images or restored.objects.get('ecg'), "restore_one_modality failed: both empty"
    print('✓ restore_one_modality ok')


# ==================== 训练状态监控回调 ====================
# 导入训练状态打印回调（如果存在）
try:
    import importlib.util
    import os
    callback_path = os.path.join(os.path.dirname(__file__), 'training_status_callback.py')
    spec = importlib.util.spec_from_file_location("training_status_callback", callback_path)
    callback_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(callback_module)
    logger.info('✅ Training status callback loaded and registered.')
except Exception as e:
    logger.warning(f'⚠️ Failed to load training status callback: {e}')



from typing import Any, Dict, Optional

from swift.llm import DatasetMeta, MessagesPreprocessor, load_dataset, register_dataset


class ECGR1Preprocessor(MessagesPreprocessor):
    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        from copy import deepcopy
        messages = deepcopy(row.get('messages', []))
        
        # 移除数据集中的 system message，让 --system 参数生效
        if messages and messages[0].get('role') == 'system':
            messages.pop(0)
        

        row['messages'] = messages
        return super().preprocess(row)

register_dataset(
    DatasetMeta(
        dataset_path='/data/jinjiarui/datasets/ECG_R1_Dataset/ecg_jsons/ECG_R1_Structured_CoT/wo_protocol/ECG_R1_Structured_CoT_RL_dataset_2k_with_solution_full.jsonl',
        dataset_name='ecg_r1_structured_cot_rl_dataset_2k',
        preprocess_func=ECGR1Preprocessor(),
        tags=['ecg', 'grpo', 'vision']))

# register_dataset(
#     DatasetMeta(
#         dataset_path='/data/jinjiarui/datasets/ECG_R1_Dataset/ecg_jsons/ECG_R1_Structured_CoT/w_protocol/ECG_R1_Structured_CoT_RL_dataset_2k_with_solution_full_with_protocol.jsonl',
#         dataset_name='ecg_r1_structured_cot_rl_dataset_2k_with_protocol',
#         preprocess_func=ECGR1Preprocessor(),
#         tags=['ecg', 'grpo', 'vision']))