# ECG-R1 集成到 vLLM 的完整实现指南

## 目标

让 ECG-R1 模型在 `swift rollout` (vLLM 后端) 中正确工作，支持：
1. ECG 数据的多模态输入
2. 训练端权重同步到 vLLM 端
3. ECG tower/projector 的推理

---

## 背景知识

### Swift RLHF 的工作模式

```
训练端 (swift rlhf)                    推理端 (swift rollout)
┌─────────────────────┐               ┌─────────────────────┐
│ ECGR1 (Transformers)│               │ ECGR1 (vLLM)        │
│ - 真实权重          │  ──权重同步──►│ - dummy 初始化      │
│ - 训练更新          │    (NCCL)     │ - 接收同步的权重    │
└─────────────────────┘               └─────────────────────┘
```

- **`load_format='dummy'`**: vLLM 启动时权重随机初始化，不从磁盘加载
- **`WeightSyncWorkerExtension`**: TRL 提供的扩展，通过 NCCL 同步权重
- **权重同步机制**: 训练端遍历 `model.named_parameters()`，发送到 vLLM 端

### 关键发现：参数名必须匹配

训练端参数名 (Transformers):
```
model.ecg_tower.transformer.resblocks.0.ln_1.weight
model.ecg_tower.transformer.resblocks.0.ln_1.bias
...
model.ecg_projector.0.weight
model.ecg_projector.0.bias
model.ecg_projector.2.weight
model.ecg_projector.2.bias
```

vLLM 的权重映射器 (`hf_to_vllm_mapper`):
```python
orig_to_new_prefix = {
    'model.visual.': 'visual.',
    'lm_head.': 'language_model.lm_head.',
    'model.language_model.': 'language_model.model.',
}
```

**问题**: `model.ecg_tower.*` 和 `model.ecg_projector.*` 没有被映射！

**解决方案**: 扩展映射器，添加 ECG 参数映射。

---

## 需要创建/修改的文件

| 文件 | 作用 | 操作 |
|------|------|------|
| `ecg_r1/vllm/ecg_r1_model.py` | vLLM 模型实现 | **新建** |
| `ecg_r1/vllm/ecg_r1_processor.py` | 多模态处理器 | **新建** |
| `ecg_r1/vllm/__init__.py` | 包初始化和插件注册 | **新建** |
| `ecg_r1/setup.py` | vLLM 插件配置 | **新建/修改** |
| `ecg_r1/my_register_v3.py` | Swift 注册 | 保持不变 |

---

## 实现步骤

### 步骤 1：创建 vLLM 模型类

**文件**: `ecg_r1/vllm/ecg_r1_model.py`

**目的**: 继承 vLLM 的 `Qwen3VLForConditionalGeneration`，添加 ECG 组件

**关键实现点**:

1. **扩展权重映射器**:
```python
hf_to_vllm_mapper = WeightsMapper(
    orig_to_new_prefix={
        "model.visual.": "visual.",
        "lm_head.": "language_model.lm_head.",
        "model.language_model.": "language_model.model.",
        # 新增 ECG 映射
        "model.ecg_tower.": "ecg_tower.",
        "model.ecg_projector.": "ecg_projector.",
    }
)
```

2. **添加 ECG 组件属性**:
```python
class ECGR1ForConditionalGeneration(Qwen3VLForConditionalGeneration):
    def __init__(self, *, vllm_config, prefix="model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        
        # ECG 组件 (结构必须与训练端一致)
        self.ecg_tower = self._build_ecg_tower()
        self.ecg_projector = self._build_ecg_projector()
```

3. **重写 `_parse_and_validate_multimodal_inputs()`**: 添加 ECG 输入解析

4. **重写 `get_multimodal_embeddings()`**: 添加 ECG embedding 计算

5. **重写 `get_input_embeddings()`**: 添加 ECG embedding 融合

6. **`load_weights()` 不需要重写**: 父类实现 + `hf_to_vllm_mapper` + `AutoWeightsLoader` 会自动处理 ECG 权重加载

---

### 步骤 2：创建多模态处理器

**文件**: `ecg_r1/vllm/ecg_r1_processor.py`

**目的**: 继承 `Qwen3VLMultiModalProcessor`，添加 ECG 数据处理

**需要重写的方法**:

| 方法 | 作用 |
|------|------|
| `_get_data_parser()` | 返回支持 ECG 的数据解析器 |
| `_hf_processor_applies_updates()` | ⚠️ **必须返回 True** |
| `_call_hf_processor()` | ⚠️ **手动展开 ECG placeholder** |
| `_get_mm_fields_config()` | 添加 `ecg_embeds` 字段配置 |
| `_get_prompt_updates()` | 添加 ECG placeholder 处理 |

#### ⚠️ 关键注意点 1: `_hf_processor_applies_updates` 必须返回 True

```python
def _hf_processor_applies_updates(self, ...):
    """始终返回 True，让 vLLM 使用 _find_mm_placeholders 查找已展开的 placeholder"""
    return True
```

**原因**：
- 当返回 `False` 时，vLLM 会调用 `_apply_prompt_updates` 进行 text-based 替换
- 但 HF Processor 已经展开了 image placeholder
- text-based 替换会导致 tokens 重复（如 744 + 743 = 1487）

#### ⚠️ 关键注意点 2: 手动展开 ECG placeholder

HF Processor 不认识 ECG placeholder，需要在 `_call_hf_processor` 中手动展开：

```python
def _call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs):
    # 检查 prompt 中是否有 ECG placeholder
    ecg_pattern = f"{ECG_START_TOKEN}{ECG_PLACEHOLDER}{ECG_END_TOKEN}"
    
    if ecg_pattern in prompt:
        # 展开: <|ecg_pad|> -> <|ecg_pad|> * tokens_per_ecg
        ecg_expanded = f"{ECG_START_TOKEN}" + ECG_PLACEHOLDER * tokens_per_ecg + f"{ECG_END_TOKEN}"
        prompt = prompt.replace(ecg_pattern, ecg_expanded)
    
    return super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)
```

#### ⚠️ 关键注意点 3: `_get_prompt_updates` 的 target 要匹配展开后的形式

```python
def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
    updates = list(super()._get_prompt_updates(...))
    
    # target 必须是展开后的形式！
    ecg_expanded_pattern = f"{ECG_START_TOKEN}" + ECG_PLACEHOLDER * tokens_per_ecg + f"{ECG_END_TOKEN}"
    
    updates.append(PromptReplacement(
        modality="ecg",
        target=ecg_expanded_pattern,  # ← 展开后的形式
        replacement=get_ecg_replacement,
    ))
    return updates
```

**关键实现**:
```python
def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
    config = super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs)
    
    # 添加 ECG 字段配置
    if 'ecg_embeds' in hf_inputs:
        config['ecg_embeds'] = MultiModalFieldConfig.batched("ecg")
    
    return config
```

---

### 步骤 3：创建数据解析器、处理信息类和虚拟输入构建器

**文件**: `ecg_r1/vllm/ecg_r1_processor.py` (同一文件)

**目的**: 让 vLLM 知道 ECG 模态的限制

#### ⚠️ 关键注意点: `ECGDataItems` 不能继承 `EmbeddingItems`

```python
# ❌ 错误做法 - 会导致 _hf_processor_applies_updates 返回 False
class ECGEmbeddingItems(EmbeddingItems):
    ...

# ✅ 正确做法 - 继承 ModalityDataItems
class ECGDataItems(ModalityDataItems):
    """ECG 数据项 - 不继承 EmbeddingItems"""
    
    def __init__(self, data: torch.Tensor):
        self.data = data
        self._modality = "ecg"
    
    @property
    def modality(self) -> str:
        return self._modality
    
    def get_count(self) -> int:
        return self.data.shape[0] if self.data.ndim >= 1 else 1
    
    def get(self, index: int) -> torch.Tensor:
        return self.data[index]
    
    def get_processor_data(self) -> Mapping[str, Any]:
        return {}  # ECG 不需要 HF Processor 处理
    
    def get_passthrough_data(self) -> Mapping[str, Any]:
        return {"ecg_embeds": self.data}
```

**原因**：
- 当 `mm_items` 中有 `EmbeddingItems` 时，`_hf_processor_applies_updates` 返回 `False`
- 这会导致 vLLM 使用错误的 text-based 替换逻辑
- 通过继承 `ModalityDataItems` 而非 `EmbeddingItems`，避免此问题

#### 处理信息类和虚拟输入构建器

```python
class ECGR1ProcessingInfo(Qwen3VLProcessingInfo):
    def get_supported_mm_limits(self):
        limits = super().get_supported_mm_limits()
        limits["ecg"] = 1  # 每个请求最多 1 个 ECG
        return limits

class ECGR1DummyInputsBuilder(Qwen3VLDummyInputsBuilder):
    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options=None):
        data = super().get_dummy_mm_data(seq_len, mm_counts, mm_options)
        if mm_counts.get("ecg", 0) > 0:
            # 生成虚拟 ECG 数据: (batch, 12 leads, 5000 samples)
            data["ecg"] = torch.zeros(mm_counts["ecg"], 12, 5000)
        return data
```

---

### 步骤 4：注册模型到 vLLM

**文件**: `ecg_r1/vllm/__init__.py`

**方式**: 使用 vLLM 插件系统

```python
def register():
    """vLLM 插件入口函数"""
    from vllm.model_executor.models import ModelRegistry
    from .ecg_r1_model import ECGR1ForConditionalGeneration
    
    ModelRegistry.register_model(
        "ECGR1ForConditionalGeneration", 
        ECGR1ForConditionalGeneration
    )
```

**文件**: `ecg_r1/setup.py`

```python
from setuptools import setup

setup(
    name="ecg_r1",
    version="0.1.0",
    packages=["ecg_r1", "ecg_r1.vllm"],
    entry_points={
        "vllm.general_plugins": [
            "ecg_r1 = ecg_r1.vllm:register",
        ],
    },
)
```

**安装**:
```bash
cd ecg_r1 && pip install -e .
```

---

### 步骤 5：处理 ECG 数据流 (✅ 调查完成)

**完整数据流路径**:

```
┌─────────────────────────────────────────────────────────────────┐
│ Swift 训练/推理端                                                │
├─────────────────────────────────────────────────────────────────┤
│ 1. ECGR1Template._pre_tokenize()                                │
│    └── <ecg> → <|ecg_start|><|ecg_pad|><|ecg_end|>              │
│                                                                 │
│ 2. ECGR1Template._encode()                                      │
│    ├── 扩展 token: 1 placeholder → 101 tokens                   │
│    ├── 加载 ECG: load_ecg() → tensor(12, 5000)                  │
│    └── 输出: {'input_ids': [...], 'ecg_features': (b,12,5000)}  │
│                                                                 │
│ 3. VllmEngine._add_request() [vllm_engine.py:320-391]           │
│    ├── mm_data['ecg'] = inputs['ecg_features']                  │
│    ├── mm_data['image'] = inputs['images']                      │
│    └── llm_inputs = {prompt_token_ids, multi_modal_data, ...}   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ vLLM 推理端                                                      │
├─────────────────────────────────────────────────────────────────┤
│ 4. ECGR1DataParser.parse_mm_data()                              │
│    └── mm_data['ecg'] → ECGDataItems(tensor)                    │
│                                                                 │
│ 5. ECGR1MultiModalProcessor._call_hf_processor()                │
│    ├── 手动展开 ECG placeholder (101 tokens)                     │
│    ├── 调用 HF Processor 处理图像                                │
│    └── 返回 BatchFeature + ecg_embeds                           │
│                                                                 │
│ 6. ECGDataItems.get_passthrough_data()                          │
│    └── {'ecg_embeds': tensor(b, 12, 5000)} → 模型               │
│                                                                 │
│ 7. Model._parse_and_validate_ecg_input()                        │
│    └── 解析 kwargs['ecg_embeds'] → ECGR1EmbeddingInputs         │
│                                                                 │
│ 8. Model._process_ecg_input()                                   │
│    ├── ecg_tower(x) → (b, 101, 768)                             │
│    ├── ecg_projector(x) → (b, 101, 4096)                        │
│    └── split → tuple of (101, 4096)                             │
│                                                                 │
│ 9. Model.get_input_embeddings()                                 │
│    ├── 分离 ECG vs Image/Video embeddings (by hidden_size)      │
│    ├── Image/Video: merge_multimodal_embeddings + deepstack     │
│    └── ECG: merge_multimodal_embeddings(ecg_token_id)           │
└─────────────────────────────────────────────────────────────────┘
```

**关键数据格式转换**:

| 位置 | 键名 | 形状 | 说明 |
|------|------|------|------|
| Template 输出 | `ecg_features` | `(batch, 12, 5000)` | 原始 ECG 信号 |
| mm_data | `ecg` | `(batch, 12, 5000)` | 传递给 vLLM |
| ECGDataItems | `data` | `(batch, 12, 5000)` | 解析后 |
| passthrough_data | `ecg_embeds` | `(batch, 12, 5000)` | 传给模型 |
| tower 后 | - | `(batch, 101, 768)` | 特征提取 |
| projector 后 | - | `(batch, 101, 4096)` | 维度映射 |
| 融合格式 | - | `tuple((101, 4096), ...)` | 每个 ECG 一个 2D tensor |

**ECG Token 数量计算**:

```python
# 环境变量
ECG_SEQ_LENGTH = 5000  # 信号长度
ECG_PATCH_SIZE = 50    # patch 大小

# Token 数量
ecg_num_patches = ECG_SEQ_LENGTH // ECG_PATCH_SIZE  # = 100
tokens_per_ecg = ecg_num_patches + 1                 # = 101 (+1 for cls token)
```

---

### 步骤 6：权重同步支持 (调查中)

#### 权重同步机制 (TRL + vLLM)

**工作流程**:

```
训练端                                    vLLM 推理端
┌─────────────────────┐                 ┌─────────────────────┐
│ Swift RLHF Trainer  │                 │ WeightSyncWorker    │
│                     │                 │ Extension           │
│ for name, param in  │                 │                     │
│   model.named_params│  ─── NCCL ───►  │ update_named_param()│
│   update_named_param│   Broadcast     │   └── load_weights()│
│   (name, param.data)│                 │                     │
└─────────────────────┘                 └─────────────────────┘
```

**关键代码位置**:

| 组件 | 文件 | 功能 |
|------|------|------|
| 训练端发送 | `trl/trainer/rloo_trainer.py:880-886` | 遍历参数，调用 update_named_param |
| vLLM 接收 | `trl/scripts/vllm_serve.py:129-153` | 接收广播，调用 load_weights |
| 权重加载 | `vllm/model_executor/models/qwen3_vl.py:1596-1603` | 使用 hf_to_vllm_mapper 加载 |

#### 参数名映射

训练端参数名使用 Transformers 格式，vLLM 模型属性名不同，需要映射：

```python
# ecg_r1_model.py
hf_to_vllm_mapper = WeightsMapper(
    orig_to_new_prefix={
        # Qwen3VL 原有映射
        "model.visual.": "visual.",
        "lm_head.": "language_model.lm_head.",
        "model.language_model.": "language_model.model.",
        # ECG 新增映射
        "model.ecg_tower.": "ecg_tower.",
        "model.ecg_projector.": "ecg_projector.",
    }
)
```

**映射示例**:

| 训练端参数名 | vLLM 属性名 |
|-------------|-------------|
| `model.ecg_tower.conv1.weight` | `ecg_tower.conv1.weight` |
| `model.ecg_tower.transformer.resblocks.0.ln_1.weight` | `ecg_tower.transformer.resblocks.0.ln_1.weight` |
| `model.ecg_projector.0.weight` | `ecg_projector.0.weight` |
| `model.ecg_projector.2.bias` | `ecg_projector.2.bias` |

#### load_weights 继承关系

```
ECGR1ForConditionalGeneration
    └── 继承 Qwen3VLForConditionalGeneration.load_weights()
            │
            └── def load_weights(self, weights):
                    loader = AutoWeightsLoader(self)
                    return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
                                                                    ↑
                                            使用子类覆盖的 hf_to_vllm_mapper
```

**关键点**:
- ✅ 子类 `hf_to_vllm_mapper` 会被父类 `load_weights()` 使用
- ✅ 不需要重写 `load_weights()` 方法
- ✅ `AutoWeightsLoader` 会自动发现 `ecg_tower` 和 `ecg_projector` 属性

#### 关键点:
- vLLM 使用 `load_format='dummy'` 启动，权重随机初始化
- 训练端通过 `update_named_param(name, weight)` 同步权重
- vLLM 端的 `load_weights()` 使用 `hf_to_vllm_mapper` 转换参数名后加载
- **ECG 组件结构必须与训练端一致**

**ECG Tower 结构** (必须匹配):
```
EcgTransformer
├── conv1: Conv1d
├── patch_dropout: Identity
├── ln_pre: LayerNorm
├── transformer: Transformer
│   └── resblocks: ModuleList (12 layers)
└── ln_post: LayerNorm
```

**ECG Projector 结构** (必须匹配):
```
Sequential
├── 0: Linear(768, 4096)
├── 1: GELU()
└── 2: Linear(4096, 4096)
```

---

### 步骤 6：mm_processor_kwargs 传递 (待实现)

**问题**: Swift rollout 时，`mm_processor_kwargs` 需要正确传递到 vLLM

**Swift 框架支持**:

1. `StdTemplateInputs` 有 `mm_processor_kwargs` 字段 (template_inputs.py:150)
2. `Template._encode_truncated()` 在 vllm 模式下会提取它 (base.py:1194-1199)
3. `VllmEngine._add_request()` 会传递给 llm_inputs (vllm_engine.py:362-364)

**当前状态**:

- ❌ ECGR1Template 没有设置 `mm_processor_kwargs`
- ✅ vLLM 端 ECGR1MultiModalProcessor 能正确处理
- ✅ 测试脚本中手动设置了 `mm_processor_kwargs`

**解决方案**:

```python
# 在 ECGR1Template._encode() 中添加
def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
    encoded = super()._encode(inputs)
    
    # 设置 mm_processor_kwargs (图像处理参数)
    factor = 32  # patch_size(16) × merge_size(2)
    max_tokens = int(os.environ.get('IMAGE_MAX_TOKEN_NUM', '768'))
    min_tokens = int(os.environ.get('IMAGE_MIN_TOKEN_NUM', '4'))
    inputs.mm_processor_kwargs = {
        'min_pixels': min_tokens * (factor ** 2),  # 4,096
        'max_pixels': max_tokens * (factor ** 2),  # 786,432
    }
    
    # ... 其他 ECG 处理 ...
    return encoded
```

**或者**在 rollout 启动时设置环境变量:

```bash
export IMAGE_MAX_TOKEN_NUM=768
export IMAGE_MIN_TOKEN_NUM=4
```

vLLM 的 HF Processor 会读取这些环境变量（如果 `patch_qwen_vl_utils` 被调用）。

---

## 核心类关系图

```
                    ┌─────────────────────────────────────┐
                    │  @MULTIMODAL_REGISTRY.register_     │
                    │  processor(ECGR1MultiModalProcessor,│
                    │  info=ECGR1ProcessingInfo,          │
                    │  dummy_inputs=ECGR1DummyInputs)     │
                    └────────────────┬────────────────────┘
                                     │
                    ┌────────────────▼────────────────────┐
                    │  ECGR1ForConditionalGeneration      │
                    │  (继承 Qwen3VLForConditionalGeneration)│
                    ├─────────────────────────────────────┤
                    │  属性:                               │
                    │  - visual                           │
                    │  - language_model                   │
                    │  - ecg_tower      ← 新增            │
                    │  - ecg_projector  ← 新增            │
                    ├─────────────────────────────────────┤
                    │  方法:                               │
                    │  + get_multimodal_embeddings()      │
                    │  + get_input_embeddings()           │
                    │  + hf_to_vllm_mapper (扩展)         │
                    │  (load_weights 继承父类即可)        │
                    └─────────────────────────────────────┘
```

---

## 验证清单

| # | 验证内容 | 预期日志 |
|---|---------|---------|
| 1 | 模型被 vLLM 识别 | `Resolved architecture: ECGR1ForConditionalGeneration` |
| 2 | 插件加载成功 | `✅ [ECG-R1 Plugin] registered` |
| 3 | ECG 组件初始化 | `✅ [ECG-R1] ECG components attached` |
| 4 | 权重同步成功 | 训练开始后无报错 |
| 5 | ECG 数据解析 | `[ECG-R1 parse_mm_data] Added ECG items` |
| 6 | ECG embeddings 计算 | `[ECG-R1 get_multimodal_embeddings] ECG processed` |
| 7 | 推理输出正常 | 生成的文本有意义（非乱码） |

---

## 实现顺序建议

1. **步骤 1** → 创建模型类骨架，添加 ECG 组件
2. **步骤 4** → 注册模型，验证能被 vLLM 识别
3. **步骤 6** → 实现权重加载逻辑，验证权重同步
4. **步骤 2-3** → 实现多模态处理器
5. **步骤 5** → 完成数据流，端到端测试

---

## 已知问题与分析

### 问题 1: ECG + Image 多模态推理失败 (2024-12-03)

**错误信息**:
```
masked_scatter_size_check: Assertion `totalElements <= srcSize` failed
```

---

## 深度调查: Qwen3VL 图像处理流程

### 1. 关键参数差异

| 参数 | 训练端 (Swift) | vLLM 端 (默认) | 说明 |
|------|---------------|---------------|------|
| `patch_size` | 16 | 16 | 固定值 |
| `spatial_merge_size` | 2 | 2 | 固定值 |
| `factor` | 32 | 32 | `patch_size × merge_size` |
| `IMAGE_MAX_TOKEN_NUM` | **768** (环境变量) | **16384** (默认) | 最大图像 tokens |
| `IMAGE_MIN_TOKEN_NUM` | **4** (环境变量) | **4** (默认) | 最小图像 tokens |
| `max_pixels` | 768 × 32² = **786,432** | 16,777,216 (默认) | `max_tokens × factor²` |
| `min_pixels` | 4 × 32² = **4,096** | 4,096 (默认) | `min_tokens × factor²` |

### 2. Swift 框架图像处理

**入口点**: `swift/llm/model/model/qwen.py::patch_qwen_vl_utils`

```python
def patch_qwen_vl_utils(vision_process):
    # 从环境变量读取参数并设置到 qwen_vl_utils.vision_process
    for key in ['image_max_token_num', 'max_pixels', ...]:
        val = get_env_args(key, type_func, default_value)
        setattr(vision_process, key.upper(), val)
```

**关键点**:
- Swift 通过 `patch_qwen_vl_utils` 设置 `IMAGE_MAX_TOKEN_NUM=768`
- `qwen_vl_utils.vision_process` 使用这个值计算 `max_pixels`
- 公式: `max_pixels = IMAGE_MAX_TOKEN_NUM × factor² = 768 × 32² = 786,432`

### 3. vLLM 图像处理

**入口点**: `vllm/model_executor/models/qwen3_vl.py`

```python
# 第 623-624 行
min_pixels=image_processor.size["shortest_edge"],
max_pixels=image_processor.size["longest_edge"],
```

**关键点**:
- vLLM 从 `image_processor.size` 读取 min/max_pixels
- 默认值: `shortest_edge=65536`, `longest_edge=16777216`
- **这些默认值巨大，基本不会缩放任何图像！**

### 4. 测试图像计算验证

**测试图像**: `mimic/p1127/p11273115/s44111511/44111511-0.png`
- 原始尺寸: 1872×1446 = 2,706,912 pixels

**训练端 (IMAGE_MAX_TOKEN_NUM=768)**:
```
max_pixels = 768 × 32² = 786,432
原始 > max_pixels → 缩放
缩放后: 992×768 = 761,856 pixels
grid: (48, 62)
tokens = 48×62/4 = 744 ✓
```

**vLLM 端 (使用默认 max_pixels)**:
```
max_pixels = 16,777,216 (默认)
原始 < max_pixels → 不缩放
未缩放: 1856×1440 = 2,672,640 pixels
grid: (90, 116)
tokens = 90×116/4 = 2610 (很大!)
```

**vLLM 端 (使用我传入的 mm_processor_kwargs)**:
```
我传入: max_pixels = 602,112 (使用错误的 factor=28)
实际处理: image_grid_thw = [1, 42, 54]
tokens = 42×54/4 = 567
```

### 5. 问题根源

1. **factor 计算错误**: 我使用 `factor=28` 而非 `factor=32`
2. **mm_processor_kwargs 部分生效**: 
   - HF processor 处理图像时使用了我传入的参数
   - 但 `input_ids` 中的 placeholder 数量可能在其他地方计算
3. **统计代码可能有问题**: 需要验证 `get_input_embeddings` 中的 token 统计

### 6. 调试脚本验证结果 (2024-12-03)

运行 `debug_qwen3vl_processing.py` 的关键发现：

**HF Processor 默认参数**:
```python
image_processor.size = {
    'longest_edge': 16,777,216,  # 巨大！约 16M pixels
    'shortest_edge': 65,536
}
```

**测试图像 (1872×1446) 的处理结果**:

| 参数 | image_grid_thw | Tokens |
|------|---------------|--------|
| 默认 (不缩放) | [1, 90, 116] | **2610** |
| max_pixels=786,432 | [1, 48, 62] | **744** |

**关键验证 - HF Processor 自身是一致的**:
```
input_ids 中 image_token_id 数量: 2610
image_grid_thw 计算的 token 数: 2610
✅ 匹配！
```

**问题根源确认**:
- vLLM 的 `mm_processor_kwargs` 可能只影响 pixel_values 处理
- **Tokenization 阶段可能在 mm_processor_kwargs 生效前完成**
- 导致 input_ids 和 embeddings 数量不匹配

### 7. 解决方案 ✅ 已实现

**最终方案：正确实现 Processor + 请求级别传递 mm_processor_kwargs**

不需要手动预处理图像！只需：

1. **`ECGDataItems` 继承 `ModalityDataItems`**（不继承 `EmbeddingItems`）
2. **`_hf_processor_applies_updates` 始终返回 `True`**
3. **在 `_call_hf_processor` 中手动展开 ECG placeholder**
4. **`_get_prompt_updates` 的 target 匹配展开后的形式**
5. **请求级别传递 `mm_processor_kwargs`**

```python
# 从环境变量计算 mm_processor_kwargs
QWEN3VL_FACTOR = 32  # patch_size(16) × merge_size(2)

def get_mm_processor_kwargs():
    max_tokens = int(os.environ.get('IMAGE_MAX_TOKEN_NUM', '768'))
    min_tokens = int(os.environ.get('IMAGE_MIN_TOKEN_NUM', '4'))
    return {
        "min_pixels": min_tokens * (QWEN3VL_FACTOR ** 2),  # e.g., 4 × 32² = 4,096
        "max_pixels": max_tokens * (QWEN3VL_FACTOR ** 2),  # e.g., 768 × 32² = 786,432
    }

# 使用方式
mm_processor_kwargs = get_mm_processor_kwargs()

text_prompt = {
    'prompt': prompt,
    'multi_modal_data': {'image': [image], 'ecg': [ecg_tensor]},
    'mm_processor_kwargs': mm_processor_kwargs,  # ← 请求级别传递
}

outputs = llm.generate([text_prompt], sampling_params)
```

**验证结果**:
```
[ECG-R1 merge] tokens in input_ids: image=744, video=0, ecg=101 ✅
```

---

## 参考资料

- vLLM 多模态支持文档: https://docs.vllm.ai/en/latest/contributing/model/multimodal/
- vLLM 模型注册文档: https://docs.vllm.ai/en/latest/contributing/model/registration/
- vLLM 插件系统: `vllm/plugins/__init__.py`
- Qwen3VL 实现: `vllm/model_executor/models/qwen3_vl.py`

