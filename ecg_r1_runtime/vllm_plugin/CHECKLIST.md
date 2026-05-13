# ECG-R1 vLLM 集成实现检查清单

## 当前状态

- [x] 步骤 1: 创建 vLLM 模型类 ✅ (2024-12-03)
- [x] 步骤 2: 创建多模态处理器 ✅ (2024-12-03)
- [x] 步骤 3: 创建处理信息类和虚拟输入构建器 ✅ (2024-12-03)
- [x] 步骤 4: 注册模型到 vLLM ✅ (2024-12-03)
- [ ] 步骤 5: 处理 ECG 数据流
- [ ] 步骤 6: 权重同步支持

---

## 步骤 1: 创建 vLLM 模型类 `ECGR1ForConditionalGeneration` ✅

**文件**: `ecg_r1/vllm/ecg_r1_model.py`

### 已实现的内容

- [x] 继承 `Qwen3VLForConditionalGeneration`
- [x] 扩展 `hf_to_vllm_mapper` 添加 ECG 参数映射
  - `"model.ecg_tower.": "ecg_tower."`
  - `"model.ecg_projector.": "ecg_projector."`
- [x] 添加 `ecg_tower` 属性
- [x] 添加 `ecg_projector` 属性
- [x] 重写 `_parse_and_validate_multimodal_inputs()` - 添加 ECG 解析
- [x] 重写 `get_multimodal_embeddings()` - 添加 ECG embedding 计算
- [x] 重写 `get_input_embeddings()` - 添加 ECG embedding 融合
- [x] ~~重写 `load_weights()`~~ - **不需要**，父类 + 正确的 mapper 就够了
- [x] 添加 `get_placeholder_str()` - ECG placeholder

### 验证 ✅

```bash
# 测试模型类是否能正确导入
python -c "from ecg_r1.vllm.ecg_r1_model import ECGR1ForConditionalGeneration; print('OK')"
# 结果: OK
```

---

## 步骤 2: 创建多模态处理器 `ECGR1MultiModalProcessor` ✅

**文件**: `ecg_r1/vllm/ecg_r1_processor.py`

### 已实现的内容

- [x] 继承 `Qwen3VLMultiModalProcessor`
- [x] 重写 `_get_data_parser()` - 支持 ECG 数据解析
- [x] **重写 `_hf_processor_applies_updates()` - 始终返回 True** ⚠️ 关键
- [x] **重写 `_call_hf_processor()` - 手动展开 ECG placeholder** ⚠️ 关键
- [x] 重写 `_get_mm_fields_config()` - 添加 ECG 字段配置
- [x] **重写 `_get_prompt_updates()` - target 匹配展开后的形式** ⚠️ 关键

### ⚠️ 关键注意点

1. **`_hf_processor_applies_updates` 必须返回 True**
   - 否则 vLLM 会使用错误的 text-based 替换，导致 tokens 重复

2. **`_call_hf_processor` 中手动展开 ECG placeholder**
   - HF Processor 不认识 ECG tokens，需要手动展开
   - `<|ecg_start|><|ecg_pad|><|ecg_end|>` → `<|ecg_start|><|ecg_pad|>×101<|ecg_end|>`

3. **`_get_prompt_updates` 的 target 必须匹配展开后的形式**
   - 不是 `<|ecg_start|><|ecg_pad|><|ecg_end|>`
   - 而是 `<|ecg_start|>` + `<|ecg_pad|>×tokens_per_ecg` + `<|ecg_end|>`

### 验证 ✅

```bash
# 测试处理器是否能正确导入
python -c "from ecg_r1.vllm.ecg_r1_processor import ECGR1MultiModalProcessor; print('OK')"
# 结果: OK
```

---

## 步骤 3: 创建数据解析器、处理信息类和虚拟输入构建器 ✅

**文件**: `ecg_r1/vllm/ecg_r1_processor.py` (同一文件)

### 已实现的内容

- [x] **`ECGDataItems` - 继承 `ModalityDataItems`** ⚠️ 关键
  - [x] 实现 `get_count()`, `get()`, `modality` 属性
  - [x] `get_processor_data()` 返回空字典
  - [x] `get_passthrough_data()` 返回 `{"ecg_embeds": self.data}`
- [x] `ECGR1DataParser` - 继承 `MultiModalDataParser`
  - [x] `_parse_ecg_data()` - 解析 ECG 数据
- [x] `ECGR1ProcessingInfo` - 继承 `Qwen3VLProcessingInfo`
  - [x] `get_supported_mm_limits()` - 添加 ECG 限制
  - [x] `get_ecg_num_tokens()` - 获取 ECG token 数量
- [x] `ECGR1DummyInputsBuilder` - 继承 `Qwen3VLDummyInputsBuilder`
  - [x] `get_dummy_text()` - 生成包含 ECG placeholder 的文本
  - [x] `get_dummy_mm_data()` - 生成虚拟 ECG 数据

### ⚠️ 关键注意点

**`ECGDataItems` 必须继承 `ModalityDataItems`，不能继承 `EmbeddingItems`！**

```python
# ❌ 错误 - 会导致 _hf_processor_applies_updates 返回 False
class ECGEmbeddingItems(EmbeddingItems):
    ...

# ✅ 正确
class ECGDataItems(ModalityDataItems):
    ...
```

**原因**：vLLM 的 `_hf_processor_applies_updates` 会检查 `mm_items` 中是否有 `EmbeddingItems`：
```python
def _hf_processor_applies_updates(self, ...):
    return not any(isinstance(items, EmbeddingItems) for items in mm_items.values())
```
如果有 `EmbeddingItems`，返回 `False`，导致使用错误的 text-based 替换逻辑。

---

## 步骤 4: 注册模型到 vLLM ✅

### 已创建的文件

- [x] `ecg_r1/vllm/__init__.py` - 包含 `register()` 函数
- [x] `ecg_r1/__init__.py` - 包初始化
- [x] `ecg_r1/setup.py` - 配置 `entry_points`

### `__init__.py` 内容

```python
def register():
    from vllm.model_executor.models import ModelRegistry
    from .ecg_r1_model import ECGR1ForConditionalGeneration
    
    ModelRegistry.register_model(
        "ECGR1ForConditionalGeneration", 
        ECGR1ForConditionalGeneration
    )
```

### `setup.py` 内容

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

### 安装和验证 ✅

```bash
cd /path/to/ecg-r1-web
pip install -e .

# 验证插件注册
python -c "
from vllm.plugins import load_general_plugins
load_general_plugins()
from vllm.model_executor.models import ModelRegistry
print('ECGR1ForConditionalGeneration' in ModelRegistry.get_supported_archs())
"
# 结果: True
```

---

## 步骤 5: 处理 ECG 数据流 ✅ 完成 (2024-12-03)

### 最终验证结果

```
[vLLM Engine DEBUG] ecg_features found! type=<class 'torch.Tensor'>, shape=torch.Size([1, 12, 5000]) ✅
[ECG-R1 merge] tokens in input_ids: image=744, video=0, ecg=101 ✅
[ECG-R1 merge] ECG merged ✅
```

### ⚠️ 关键发现: vLLM 模式下 `_encode()` 被跳过

**问题根因**:

Swift 框架在 vLLM 模式下的 `_encode_truncated()` 方法直接调用基类：

```python
# swift/llm/template/base.py:1192
if self.mode in {'vllm', 'lmdeploy', 'sglang'}:
    encoded = Template._encode(self, inputs)  # ← 跳过子类 _encode()!
```

这导致 `ECGR1Template._encode()` 完全不会被调用，ECG 数据流中断。

**解决方案**: 重写 `_encode_truncated()` 方法

```python
def _encode_truncated(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
    """重写以确保 vLLM 模式也处理 ECG"""
    encoded = super()._encode_truncated(inputs)
    
    # vLLM 模式下父类跳过了子类 _encode，手动处理 ECG
    if self.mode in {'vllm', 'lmdeploy', 'sglang'}:
        encoded = self._postprocess_ecg(encoded, inputs)
    
    return encoded

def _postprocess_ecg(self, encoded, inputs):
    """抽取为独立方法，供 _encode 和 _encode_truncated 调用"""
    # 设置 mm_processor_kwargs
    mm_processor_kwargs = {...}
    inputs.mm_processor_kwargs = mm_processor_kwargs
    encoded['mm_processor_kwargs'] = mm_processor_kwargs  # 确保 vLLM 模式也生效
    
    # ECG token 扩展和数据加载...
```

### 完整数据流

```
┌─────────────────────────────────────────────────────────────────┐
│ Swift 端                                                        │
├─────────────────────────────────────────────────────────────────┤
│ 1. ECGR1Template._encode_truncated() [重写]                     │
│    ├── super()._encode_truncated()                              │
│    └── _postprocess_ecg()                                       │
│        ├── ECG token 扩展: 1 → 101                              │
│        ├── ECG 数据加载: load_ecg() → tensor(12, 5000)         │
│        ├── mm_processor_kwargs 设置                             │
│        └── 输出: {'input_ids', 'ecg_features', ...}            │
│                                                                 │
│ 2. VllmEngine._add_request()                                    │
│    ├── mm_data['ecg'] = inputs['ecg_features']                 │
│    └── llm_inputs['mm_processor_kwargs'] = ...                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ vLLM 端                                                          │
├─────────────────────────────────────────────────────────────────┤
│ 3. ECGR1DataParser → ECGDataItems                               │
│ 4. ECGR1MultiModalProcessor._call_hf_processor()                │
│ 5. ECGR1ForConditionalGeneration._process_ecg_input()           │
│    └── ecg_tower → ecg_projector → embeddings                   │
│ 6. get_input_embeddings() → merge_multimodal_embeddings         │
└─────────────────────────────────────────────────────────────────┘
```

### 验证清单

- [x] Swift 端 `ecg_features` 传递 ✅
- [x] vLLM 解析 `ecg` 数据 ✅
- [x] ECG placeholder 展开 (101 tokens) ✅
- [x] ECG embeddings 计算 ✅
- [x] ECG + Image 融合 ✅
- [x] mm_processor_kwargs 传递 ✅
- [x] **_encode_truncated 重写** ✅ 关键修复
    }
    return encoded
```

方案 B - 在 vLLM Engine 初始化时设置默认值:
- 在 rollout 脚本中设置 `mm_processor_kwargs`

### 验证命令

```bash
# 运行 rollout 测试
bash shells/rlhf_train/ecg-r1-8b-rollout.sh

# 运行测试脚本
python ecg_r1/vllm_plugin/test_vllm_vs_pt.py
```

---

## 步骤 6: 权重同步支持 (调查中)

### 调查总结 (2024-12-03)

#### 权重同步机制概述

```
┌─────────────────────────────────────────────────────────────────┐
│ 训练端 (Swift RLHF Trainer)                                     │
├─────────────────────────────────────────────────────────────────┤
│ 1. model.named_parameters() 遍历所有参数                        │
│ 2. vllm_client.update_named_param(name, weight)                │
│    ├── name: "model.ecg_tower.conv1.weight"                    │
│    └── weight: tensor(...)                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                   NCCL Broadcast (PyNcclCommunicator)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ vLLM 端 (WeightSyncWorkerExtension)                             │
├─────────────────────────────────────────────────────────────────┤
│ 3. update_named_param(name, dtype, shape)                       │
│ 4. pynccl_comm.broadcast(weight, src=client_rank)              │
│ 5. model_runner.model.load_weights([(name, weight)])           │
│    └── 使用 hf_to_vllm_mapper 转换参数名                        │
└─────────────────────────────────────────────────────────────────┘
```

#### TRL 训练器权重同步代码

**文件**: `trl/trainer/rloo_trainer.py` (或 `online_dpo_trainer.py`)

```python
# 遍历模型参数，发送到 vLLM
for name, param in module.named_parameters():
    name = self._fix_param_name_to_vllm(name)
    if self.vllm_mode == "server":
        self.vllm_client.update_named_param(name, param.data)  # ← 关键
```

#### vLLM 端权重接收

**文件**: `trl/scripts/vllm_serve.py`

```python
class WeightSyncWorkerExtension:
    def update_named_param(self, name: str, dtype: str, shape: Sequence[int]):
        # 接收广播的权重
        weight = torch.empty(shape, dtype=dtype, device=self.device)
        self.pynccl_comm.broadcast(weight, src=self.client_rank)
        
        # 加载到模型
        self.model_runner.model.load_weights([(name, weight)])  # ← 关键
```

#### vLLM 模型 load_weights

**文件**: `vllm/model_executor/models/qwen3_vl.py`

```python
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    loader = AutoWeightsLoader(self, skip_prefixes=...)
    return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)  # ← 使用映射器
```

#### ECG-R1 权重映射

**文件**: `ecg_r1/vllm_plugin/ecg_r1_model.py`

```python
class ECGR1ForConditionalGeneration(Qwen3VLForConditionalGeneration):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # 原有 Qwen3VL 映射
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
            # 新增 ECG 映射
            "model.ecg_tower.": "ecg_tower.",
            "model.ecg_projector.": "ecg_projector.",
        }
    )
```

#### 参数名转换流程

| 阶段 | 参数名 |
|------|--------|
| 训练端 (Transformers) | `model.ecg_tower.conv1.weight` |
| 发送到 vLLM | `model.ecg_tower.conv1.weight` |
| hf_to_vllm_mapper | `model.ecg_tower.` → `ecg_tower.` |
| vLLM 模型属性 | `ecg_tower.conv1.weight` |

#### 需要验证的点

- [x] `hf_to_vllm_mapper` 包含 ECG 映射 ✅ (已在 ecg_r1_model.py 中定义)
- [ ] vLLM 端 `ecg_tower` 和 `ecg_projector` 属性存在
- [ ] 权重同步时参数能正确加载
- [ ] 模型结构与训练端完全一致

#### ECG 组件结构要求

**ECG Tower** (必须与训练端一致):
```
EcgTransformer
├── conv1: Conv1d(12, 768, kernel_size=50, stride=50)
├── patch_dropout: Identity
├── ln_pre: LayerNorm(768)
├── transformer: Transformer
│   └── resblocks: ModuleList (12 layers)
│       └── ResidualAttentionBlock
│           ├── ln_1: LayerNorm
│           ├── attn: MultiheadAttention
│           ├── ls_1: Identity
│           ├── ln_2: LayerNorm
│           ├── mlp: Sequential
│           └── ls_2: Identity
└── ln_post: LayerNorm(768)
```

**ECG Projector** (必须与训练端一致):
```
Sequential
├── 0: Linear(768, 4096)
├── 1: GELU()
└── 2: Linear(4096, 4096)
```

#### 潜在问题

1. **load_weights 是否被继承?**
   - ECGR1ForConditionalGeneration 继承自 Qwen3VLForConditionalGeneration
   - 父类的 load_weights 使用 AutoWeightsLoader 和 hf_to_vllm_mapper
   - 子类覆盖了 hf_to_vllm_mapper，应该能自动工作

2. **ECG 组件是否被 AutoWeightsLoader 识别?**
   - AutoWeightsLoader 通过 `self.named_parameters()` 遍历模型
   - 需要确保 `ecg_tower` 和 `ecg_projector` 是 `nn.Module` 类型

3. **权重同步时的设备问题?**
   - vLLM 在 GPU 上运行
   - 需要确保权重被正确放到 GPU 上

### 下一步行动

1. [ ] 验证 ECGR1ForConditionalGeneration.load_weights 继承正确
2. [ ] 测试权重同步流程 (启动 RLHF 训练)
3. [ ] 检查 ECG 组件在 vLLM 端的初始化

---

## 步骤 6: 权重同步支持

### 需要验证的内容

- [ ] ECG tower 结构与训练端一致
- [ ] ECG projector 结构与训练端一致
- [ ] 参数名映射正确
- [ ] 权重同步时无报错

### 训练端参数名格式

```
model.ecg_tower.conv1.weight
model.ecg_tower.conv1.bias
model.ecg_tower.ln_pre.weight
model.ecg_tower.ln_pre.bias
model.ecg_tower.transformer.resblocks.{0-11}.*
model.ecg_tower.ln_post.weight
model.ecg_tower.ln_post.bias
model.ecg_projector.0.weight
model.ecg_projector.0.bias
model.ecg_projector.2.weight
model.ecg_projector.2.bias
```

### vLLM 端参数名格式 (映射后)

```
ecg_tower.conv1.weight
ecg_tower.conv1.bias
ecg_tower.ln_pre.weight
ecg_tower.ln_pre.bias
ecg_tower.transformer.resblocks.{0-11}.*
ecg_tower.ln_post.weight
ecg_tower.ln_post.bias
ecg_projector.0.weight
ecg_projector.0.bias
ecg_projector.2.weight
ecg_projector.2.bias
```

---

## 端到端验证

```bash
# 1. 启动 rollout 服务
bash shells/rlhf_train/ecg-r1-8b-rollout.sh

# 2. 查看日志，确认以下内容
# - ✅ [ECG-R1 Plugin] registered
# - Resolved architecture: ECGR1ForConditionalGeneration
# - ✅ [ECG-R1] ECG components attached

# 3. 在另一个终端启动训练
bash shells/rlhf_train/ecg-r1-8b-dapo.sh

# 4. 观察是否有权重同步错误
# 5. 观察生成的文本是否正常
```

---

## 常见问题

### Q1: `ValueError: Model architectures ['ECGR1ForConditionalGeneration'] are not supported`

**原因**: 模型未注册到 vLLM

**解决**: 
1. 确保 `setup.py` 中的 `entry_points` 配置正确
2. 运行 `pip install -e .` 重新安装
3. 检查 `register()` 函数是否正确实现

### Q2: 权重同步时参数找不到

**原因**: 参数名不匹配

**解决**: 
1. 检查 `hf_to_vllm_mapper` 是否包含 ECG 映射
2. 确保 ECG 组件结构与训练端完全一致

### Q3: CUDA 错误 `masked_scatter` (2024-12-03 当前问题)

**错误信息**:
```
masked_scatter_size_check: Assertion `totalElements <= srcSize` failed
```

**原因**: `input_ids` 中的 placeholder token 数量 ≠ 实际 embedding 数量

**深度调查结论** (详见 IMPLEMENTATION_GUIDE.md):

| 指标 | 训练端 | vLLM (期望) | vLLM (实际) |
|------|--------|------------|------------|
| patch_size | 16 | 16 | 16 ✅ |
| factor | 32 | 32 | 28 ❌ (我的计算错误) |
| IMAGE_MAX_TOKEN_NUM | 768 | 768 | 16384 (默认) |
| max_pixels | 786,432 | 786,432 | 602,112 (我用错误 factor 算的) |
| Image tokens | 744 | 744 | 567 (实际处理) |

**问题根源**:
1. **我的 factor 计算错误**: 使用 `28` 而非 `32`
2. **vLLM 默认参数巨大**: `max_pixels=16,777,216`，基本不缩放
3. **mm_processor_kwargs 生效不完整**: HF processor 使用了我的参数，但其他地方可能没有

---

### Q4: 图像处理流程对比

**Swift 训练端**:
```
1. patch_qwen_vl_utils() 设置 IMAGE_MAX_TOKEN_NUM=768
2. qwen_vl_utils.vision_process 使用此值
3. max_pixels = 768 × 32² = 786,432
4. smart_resize 缩放图像
5. 计算 image_grid_thw → tokens = 744
```

**vLLM 推理端**:
```
1. 从 image_processor.size 读取参数
2. 默认 longest_edge=16,777,216 (巨大!)
3. mm_processor_kwargs 可传入自定义参数
4. 但需要确保所有阶段都使用相同参数
```

**验证结果**:
- [x] ~~smart_resize 计算是否与训练端一致~~ → **factor 计算错误 (28 vs 32)**
- [ ] mm_processor_kwargs 是否在 tokenization 阶段生效 → **需要继续调查**
- [x] ~~vLLM 是否有单独的图像尺寸配置~~ → **是的，从 image_processor.size 读取**

---

### Q5: 调试脚本验证结果 (2024-12-03)

运行 `debug_qwen3vl_processing.py` 确认了问题根源：

**HF Processor 默认值**:
```
longest_edge: 16,777,216 (巨大，约 16M pixels)
shortest_edge: 65,536
```

**测试图像 (1872×1446)**:
| 配置 | image_grid_thw | Tokens |
|------|---------------|--------|
| 默认 | [1, 90, 116] | 2610 |
| max_pixels=786,432 | [1, 48, 62] | 744 |

**关键验证**:
```
HF Processor 自身是一致的！
input_ids 中 image tokens: 2610
image_grid_thw 计算 tokens: 2610 ✅ 匹配
```

**结论**:
- `mm_processor_kwargs` 可能只影响 pixel_values 处理
- Tokenization 阶段可能在 `mm_processor_kwargs` 生效前完成
- 需要在传入 vLLM 前预处理图像

---

### Q6: vLLM 原生处理验证 (2024-12-03)

**关键发现**: vLLM + Qwen3VL 原生处理是**完全正确的**！

```
debug_vllm_processing.py 测试结果:

[HF Processor 测试]
  默认参数: 2610 tokens ✅ 匹配
  自定义 max_pixels=786,432: 744 tokens ✅ 匹配

[vLLM Engine 测试]  
  mm_processor_kwargs: {'min_pixels': 4096, 'max_pixels': 786432}
  image_token_id 数量: 744 ✅ 匹配期望值！
```

**结论**: 
- ❌ **不需要**自己实现 `smart_resize`
- ❌ **不需要**预处理图像
- ✅ 只需正确传递 `mm_processor_kwargs`
- ✅ 让 vLLM 的 HF Processor 自动处理

---

## Q7: ECGR1MultiModalProcessor 分析

### 当前实现检查

**`_call_hf_processor` 方法** (ecg_r1_processor.py:194-236):
```python
def _call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs):
    mm_data = dict(mm_data)
    ecg_data = mm_data.pop("ecg", None)  # 提取 ECG
    
    # ✅ 正确: 调用父类处理 image/video
    processed = super()._call_hf_processor(
        prompt=prompt,
        mm_data=mm_data,
        mm_kwargs=mm_kwargs,  # ✅ mm_processor_kwargs 正确传递
        tok_kwargs=tok_kwargs,
    )
    
    # 添加 ECG 数据到输出
    if ecg_data is not None:
        processed["ecg_embeds"] = ...
    
    return processed
```

**分析**: `_call_hf_processor` 实现是**正确的**:
- ✅ 正确传递 `mm_kwargs` (包含 min_pixels, max_pixels)
- ✅ 正确调用父类处理图像
- ✅ 正确添加 ECG 数据

### 潜在问题

1. **测试脚本中的冗余代码**:
   - `smart_resize_for_qwen3vl` - 不需要
   - `resize_image_for_qwen` - 不需要
   - 手动计算 token 数 - 不需要

2. **传参方式可能有问题**:
   - 需要确认 `mm_processor_kwargs` 如何传递给 vLLM

---

## Q8: 图像处理正确方案

### ❌ 错误做法 (当前测试脚本)

```python
# 不需要！
def smart_resize_for_qwen3vl(height, width, max_tokens):
    # ... 手动计算 ...

# 不需要！
image = resize_image_for_qwen(image_obj, max_tokens=768)
```

### ✅ 正确做法

```python
# 1. 直接使用原始图像
image_obj = Image.open(image_path).convert('RGB')

# 2. 从环境变量计算 mm_processor_kwargs
QWEN3VL_FACTOR = 32  # patch_size(16) × merge_size(2)
max_tokens = int(os.environ.get('IMAGE_MAX_TOKEN_NUM', '768'))
min_tokens = int(os.environ.get('IMAGE_MIN_TOKEN_NUM', '4'))
mm_processor_kwargs = {
    "min_pixels": min_tokens * (QWEN3VL_FACTOR ** 2),  # e.g., 4 × 32² = 4,096
    "max_pixels": max_tokens * (QWEN3VL_FACTOR ** 2),  # e.g., 768 × 32² = 786,432
}

# 3. 构建输入
prompt_input = {
    "prompt": prompt,
    "multi_modal_data": {
        "image": [image_obj],  # 原始图像，不预处理
        "ecg": [ecg_tensor],
    },
    "mm_processor_kwargs": mm_processor_kwargs,
}

# 4. vLLM 会自动:
#    - 使用 mm_processor_kwargs 中的参数
#    - 调用 HF Processor 处理图像
#    - 同时处理 tokenization 和 image processing
#    - 保证 input_ids 和 embeddings 数量一致
```

---

## Q9: ECG-R1 测试结果 (2024-12-03)

### 测试结果

| 测试 | 结果 | 说明 |
|------|------|------|
| 仅 ECG 推理 | ✅ | ECG tokens=101, 融合成功 |
| ECG + Image 推理 | ❌ | image tokens 不匹配 |

### 关键日志

```
[ECG-R1 merge] tokens in input_ids: image=1487, video=0, ecg=101
[ECG-R1 merge] embedding[1] shape: torch.Size([744, 16384])
mm_position=PlaceholderRange(offset=102, length=744, ...)
image_grid_thw: tensor([ 1, 48, 62])  # 48*62/4 = 744 ✓
```

### 问题分析

**纯 Qwen3VL** (debug_vllm_processing.py):
- `mm_processor_kwargs` 正确传递 ✅
- `input_ids` 中 744 个 image tokens ✅

**ECG-R1** (test_ecg_multimodal.py):
- `mm_processor_kwargs` 传递给 Engine ✅
- `image_grid_thw` 正确 (744) ✅
- `mm_position.length` 正确 (744) ✅
- **但 `input_ids` 中有 1487 个 image tokens** ❌

**根因确认** ✅:

问题出在 vLLM 的 `_hf_processor_applies_updates` 方法：

```python
def _hf_processor_applies_updates(self, ...):
    return not any(
        isinstance(items, (EmbeddingItems, DictEmbeddingItems))
        for items in mm_items.values())
```

当 `mm_items` 中有 `EmbeddingItems`（如我们的 `ECGEmbeddingItems`）时，`is_update_applied=False`！

**流程对比**:

| 场景 | is_update_applied | Prompt tokenization | 结果 |
|------|-------------------|---------------------|------|
| 只有图像 | `True` | HF Processor 直接处理，使用 `mm_processor_kwargs` | ✅ 744 tokens |
| 有 ECG | `False` | `_apply_hf_processor_text_only` 使用**空** kwargs | ❌ 1487 tokens |

**关键代码** (vllm/multimodal/processing.py:1489):
```python
def _apply_hf_processor_text_only(self, prompt_text, tokenization_kwargs):
    prompt_ids, _, _ = self._apply_hf_processor_text_mm(
        prompt_text=prompt_text,
        mm_items=MultiModalDataItems({}),
        hf_processor_mm_kwargs={},  # <-- 空的！使用默认参数
        tokenization_kwargs=tokenization_kwargs,
    )
```

**验证实验**:
```bash
# 只用图像 (不用 ECG) - ECG-R1 模型
[ECG-R1 merge] tokens in input_ids: image=744, video=0, ecg=0  ✅

# 同时使用 ECG + Image - ECG-R1 模型  
[ECG-R1 merge] tokens in input_ids: image=1487, video=0, ecg=101  ❌
```

---

## Q10: 调查总结 (2024-12-03)

### 🔍 验证实验

| # | 测试场景 | 模型 | 结果 | image tokens |
|---|---------|------|------|--------------|
| 1 | 纯 Qwen3VL + 图像 | Qwen3VL | ✅ | 744 |
| 2 | ECG-R1 + 只图像 | ECG-R1 | ✅ | 744 |
| 3 | ECG-R1 + 只 ECG | ECG-R1 | ✅ | 0 (ecg=101) |
| 4 | ECG-R1 + ECG + 图像 | ECG-R1 | ❌ | 1487 |

### 🎯 根因

**核心问题**: `ECGEmbeddingItems` 继承自 `EmbeddingItems`

```python
# ecg_r1_processor.py (当前实现)
class ECGEmbeddingItems(EmbeddingItems):  # <-- 问题所在
    ...
```

**vLLM 判断逻辑** (vllm/multimodal/processing.py:1437):
```python
def _hf_processor_applies_updates(self, ...):
    return not any(
        isinstance(items, (EmbeddingItems, DictEmbeddingItems))
        for items in mm_items.values())
    # ECGEmbeddingItems 是 EmbeddingItems → 返回 False!
```

**结果**:
- `is_update_applied=False`
- vLLM 调用 `_apply_hf_processor_text_only`
- 使用**空** `hf_processor_mm_kwargs={}`
- 图像 placeholder 按默认参数展开 → 1487 tokens

### ✅ 解决方案

**修改 `ECGEmbeddingItems` 不继承 `EmbeddingItems`**:

```python
# 修改后
class ECGDataItems(ModalityDataItems):
    """ECG 数据项 - 不继承 EmbeddingItems"""
    
    def __init__(self, data):
        super().__init__(data, "ecg")
    
    def get_processor_data(self) -> Mapping[str, Any]:
        return {}  # ECG 不需要 HF Processor 处理
    
    def get_passthrough_data(self) -> Mapping[str, Any]:
        return {"ecg_embeds": self.data}
```

**效果**:
- `_hf_processor_applies_updates` 返回 `True`
- vLLM 直接使用 HF Processor 处理 prompt + 图像
- 使用 `mm_processor_kwargs` 中的 `max_pixels`
- 图像 placeholder 正确展开 → 744 tokens

---

## Q11: 下一步行动

### 已完成 ✅

1. [x] **修改 `ECGEmbeddingItems` → `ECGDataItems`** (不继承 EmbeddingItems)
2. [x] **在 `_call_hf_processor` 中手动展开 ECG placeholder**
3. [x] **`_hf_processor_applies_updates` 始终返回 True**
4. [x] **验证 ECG + Image 推理**

### 待完成

1. [ ] 测试 swift rollout 流程

---

## Q12: 最终解决方案总结 (2024-12-03)

### 问题

当同时使用 ECG 和 Image 时，image token 数量不匹配（1487 vs 744）。

### 根因

1. 当 ECG 使用 `EmbeddingItems` 时，`_hf_processor_applies_updates` 返回 `False`
2. 导致 vLLM 调用 `_apply_prompt_updates` 进行 text-based 替换
3. HF Processor 已经展开了 image placeholder (744 tokens)
4. text-based 替换只替换了一部分，导致 744 + 743 = 1487 tokens

### 解决方案

1. **`ECGDataItems` 继承 `ModalityDataItems`**（不继承 `EmbeddingItems`）
   - 确保 `_hf_processor_applies_updates` 不被 ECG 影响

2. **在 `_call_hf_processor` 中手动展开 ECG placeholder**
   - HF Processor 不认识 ECG placeholder，需要手动展开
   ```python
   ecg_pattern = f"{ECG_START_TOKEN}{ECG_PLACEHOLDER}{ECG_END_TOKEN}"
   ecg_expanded = f"{ECG_START_TOKEN}" + ECG_PLACEHOLDER * tokens_per_ecg + f"{ECG_END_TOKEN}"
   prompt = prompt.replace(ecg_pattern, ecg_expanded, 1)
   ```

3. **`_get_prompt_updates` 中的 target 匹配展开后的形式**
   ```python
   ecg_expanded_pattern = f"{ECG_START_TOKEN}" + ECG_PLACEHOLDER * tokens_per_ecg + f"{ECG_END_TOKEN}"
   ```

4. **`_hf_processor_applies_updates` 始终返回 True**
   - 让 vLLM 使用 `_find_mm_placeholders` 查找已展开的 placeholder
   - 不触发 `_apply_prompt_updates` 的错误替换逻辑

### 验证结果

```
[ECG-R1 merge] tokens in input_ids: image=744, video=0, ecg=101 ✅
[ECG-R1 merge] embedding[0] shape: torch.Size([101, 4096])  # ECG
[ECG-R1 merge] embedding[1] shape: torch.Size([744, 16384]) # Image (deepstack)
Done!
