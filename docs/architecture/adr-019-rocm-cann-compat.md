# ADR-019: NPU CANN & AMD ROCm 生态兼容（不自研 kernel）

- **状态**：Accepted
- **日期**：2026-07-25
- **决策者**：Verse 框架作者（CometFuture / CUI-hyg）
- **相关规范**：[`/workspace/.trae/specs/part5k1.3-bugfix-stability-upgrade/spec.md`](../../../.trae/specs/part5k1.3-bugfix-stability-upgrade/spec.md)
- **前置 ADR**：[ADR-001 CPU 优先](adr-001-cpu-first.md)（零重型依赖原则）、[ADR-005 GPU/NPU 后端抽象](adr-005-gpu-npu-backend.md)（`DeviceBackend` + `TorchBackend` 委托）
- **相关 ADR**：[ADR-014 双模型并行 small/mate](adr-014-dual-model-small-mate.md)（双模型训练需支持 ROCm / NPU 设备）、[ADR-017 .vn v2 断点续训](adr-017-vn-v2-resume.md)（断点续训跨设备恢复）

## 上下文

ADR-005 已落地 `DeviceBackend` 抽象（`NumpyBackend` 默认 + `TorchBackend` PyTorch 委托），支持 `cpu` / `cuda` / `mps` / `npu` 四种设备字符串，NPU 走 `torch_npu` 扩展。但 Part5K1.3 在落地"自研原生架构稳定可用"承诺时，暴露出 3 个生态兼容短板：

1. **AMD ROCm 未显式支持**：`device.py._parse_device` 仅识别 `cpu` / `cuda` / `mps` / `npu`，AMD GPU（ROCm 生态）虽通过 PyTorch ROCm build 暴露为 `cuda` API（HIP-on-ROCm），但用户无法用 `--device rocm` 显式声明设备类型，运维诊断困难（无法区分 NVIDIA CUDA 与 AMD ROCm）；ROCm 环境下 `--device rocm` 抛 `ValueError`，用户被迫用 `--device cuda`，与硬件实际不符。
2. **CANN 版本探测缺失**：`device.py` 仅检测 `torch_npu` 是否可用（`has_torch_npu()`），无 CANN 版本探测 API；NPU 训练故障时（autocast 失败 / 算子不兼容 / 显存异常）运维无法快速判断 CANN 版本是否匹配，需要额外 `npu-smi` / `cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg` 命令排查，效率低。
3. **autocast NPU 路径无诊断日志**：`backend_torch.py.autocast` 在 NPU 路径首次启用时不打印任何版本信息，NPU 训练报错时无法从日志判断 CANN / PyTorch / torch_npu 版本组合，回溯困难；同时 ROCm 路径下 `device_type="rocm"` 未映射到 `torch.autocast` 接受的 `"cuda"`，PyTorch 不识别 `"rocm"` 字符串直接抛 `RuntimeError`。
4. **NPU 显存管理 API 兼容性差**：`get_memory_info("npu")` 仅走 `torch.npu.memory_allocated` / `torch.npu.mem_get_info`，但部分 CANN 版本 API 名差异（`memory_allocated` vs `mem_get_info`），导致版本不匹配时显存查询失败，VMPC V2.0 的多空间缓存无法触发自动卸载。

同时必须保持 ADR-005 的核心承诺：**无 PyTorch 环境下 CPU 路径完全不变**（向后兼容）；**不自研 ROCm / CANN kernel**（所有 GPU/NPU 计算走 PyTorch 原生 + `torch_npu`）。

## 决策

**`device.py._parse_device` 新增 `"rocm"` / `"rocm:0"` 识别（等价 `"cuda"`，HIP-on-ROCm）；新增 `has_rocm` / `has_cann` / `get_rocm_version` / `get_cann_version` 探测 API；`backend_torch.py.TorchBackend` 接受 `"rocm"` device 字符串内部映射到 `torch.device("cuda")`；`autocast` 在 ROCm 路径映射 `device_type="cuda"`，在 NPU 路径首次启用时打印 CANN + PyTorch + torch_npu 版本日志；`spark/run.py --device` 支持 `rocm` / `rocm:0`。**

具体含义：

1. **`device.py._parse_device` 设备字符串扩展**：

   - 新增 `"rocm"` / `"rocm:0"` 识别，等价于 `"cuda"` / `"cuda:0"`（HIP-on-ROCm 走 PyTorch `cuda` 路径）
   - `_parse_device("rocm")` 返回 `("rocm", None)`（保留原字符串用于诊断）
   - `_parse_device("rocm:0")` 返回 `("rocm", 0)`
   - `get_backend("rocm:0")` 返回 `TorchBackend(device="cuda:0")`（内部映射，对外 `device_type` 保留 `"rocm"` 用于诊断）

2. **ROCm / CANN 探测 API**（`device.py`）：

   - `has_rocm() -> bool`：检测当前 PyTorch 是否为 ROCm build（`torch.version.hip is not None`）；无 PyTorch 环境返回 `False`，不抛异常（lazy 检测）
   - `get_rocm_version() -> Optional[str]`：返回 ROCm 版本字符串（如 `"6.2.0"`）；非 ROCm build 返回 `None`
   - `has_cann() -> bool`：检测 `torch_npu` 可用 + CANN 版本可读（`torch_npu.version` 或 `torch_npu.cann_version`）；无 `torch_npu` 返回 `False`
   - `get_cann_version() -> Optional[str]`：返回 CANN 版本字符串（如 `"8.0.0"`）；无 CANN 返回 `None`

3. **`backend_torch.py.TorchBackend` 兼容**：

   - `_torch_device` 接受 `"rocm"` / `"rocm:0"`，内部映射到 `torch.device("cuda")` / `torch.device("cuda:0")`（PyTorch 不识别 `"rocm"` 字符串，但 ROCm build 通过 HIP 暴露 cuda API）
   - `TorchBackend.__init__` 接受 `"rocm"` device 字符串，`_device_str` 保留原值（如 `"rocm:0"`，用于诊断），`_torch_device` 映射到 `cuda`
   - `device_type` 属性：`_device_str` 以 `"rocm"` 开头时返回 `"rocm"`，否则按原逻辑返回 `"cpu"` / `"cuda"` / `"npu"` / `"mps"`

4. **`autocast` 上下文管理器兼容 + 日志**：

   - **ROCm 路径**：`device_type="rocm"` 等价 `device_type="cuda"`（PyTorch ROCm build 原生支持 fp16 autocast via HIP）；内部把 `"rocm"` 映射为 `"cuda"` 传给 `torch.autocast`
   - **NPU 路径首次启用日志**：`autocast` 在 NPU 路径首次启用时打印 CANN 版本 + PyTorch 版本 + torch_npu 版本日志：
     ```
     [autocast] NPU CANN version=8.0.0, torch=2.4.0, torch_npu=2.4.0
     ```
     仅首次打印（模块级 `_npu_autocast_logged` 标记），避免每个 autocast 块都刷日志
   - **CPU 路径不变**：`autocast` 在 CPU 下仍为 no-op（不破坏 ADR-001）

5. **`device.py` 显存管理兼容**：

   - `empty_cache("rocm")` / `get_memory_info("rocm")` 走 `torch.cuda.empty_cache()` / `torch.cuda.mem_get_info()`（PyTorch ROCm build 已通过 HIP 暴露 cuda API）
   - `get_memory_info("npu")` 增加 `torch_npu.npu.mem_get_info` 兜底（部分 CANN 版本 API 名差异：`memory_allocated` / `mem_get_info`）；优先尝试 `mem_get_info`，失败时降级到 `memory_allocated` 估算

6. **`spark/run.py` CLI 兼容**：

   - `--device` 参数支持 `"rocm"` / `"rocm:0"` / `"npu"` / `"cuda"` / `"cpu"` / `"mps"`
   - 启动时打印设备信息（device type + 版本 + 显存），调用 `has_rocm` / `get_rocm_version` / `has_cann` / `get_cann_version`：
     ```
     [device] type=rocm, rocm_version=6.2.0, torch=2.4.0+rocm6.2, mem_total=16GB
     [device] type=npu, cann_version=8.0.0, torch=2.4.0, torch_npu=2.4.0, mem_total=32GB
     ```

7. **不自研 kernel 约束（核心）**：

   - **不自研 ROCm kernel**：所有 GPU 计算走 PyTorch 原生（MIOpen / rocBLAS / FlashAttention-ROCm），HIP-on-ROCm 复用 cuda 路径
   - **不自研 CANN kernel**：NPU 计算走 `torch_npu`（HCCL / hccl_kernel）
   - 仅做 device 字符串识别 + autocast 适配 + 版本探测 + 显存管理兜底

## 后果

### 优点

- **AMD ROCm 显式支持**：用户可用 `--device rocm` / `--device rocm:0` 显式声明 AMD GPU，运维诊断清晰（`device_type == "rocm"` 与 NVIDIA `cuda` 区分）；ROCm 环境下无需伪装成 `cuda`。
- **CANN 版本诊断闭环**：`has_cann` / `get_cann_version` + autocast 首次启用日志，NPU 训练故障时从日志直接判断 CANN / PyTorch / torch_npu 版本组合，无需额外命令排查。
- **不自研 kernel**：所有 GPU/NPU 计算走 PyTorch 原生 + `torch_npu`，避免维护成本与正确性风险；与 ADR-005 "委托 PyTorch / 不自研 kernel"原则一致。
- **CPU 路径不变**：无 PyTorch 环境下 `has_rocm` / `has_cann` 返回 `False`，`_parse_device("rocm")` 仅做字符串识别不触发 torch import；CPU 路径完全不变（向后兼容 ADR-001）。
- **autocast ROCm 兼容**：`device_type="rocm"` 内部映射 `"cuda"`，PyTorch ROCm build 原生支持 fp16 autocast via HIP；用户代码 `with autocast(device="rocm", dtype=torch.float16):` 直接可用。
- **NPU 显存管理兜底**：`get_memory_info("npu")` 多 API 兜底，CANN 版本差异不再导致显存查询失败；VMPC V2.0 的多空间缓存可正常触发自动卸载。
- **CLI 设备信息透明**：`spark/run.py` 启动时打印设备 type + 版本 + 显存，用户启动即可确认环境，无需额外诊断脚本。

### 缺点

- **`"rocm"` 字符串内部映射增加隐式行为**：用户传 `"rocm:0"` 后 `TorchBackend._device_str == "rocm:0"` 但 `_torch_device == torch.device("cuda:0")`，需文档明确"对外诊断用 rocm，内部计算用 cuda"。
- **CANN 版本探测依赖 `torch_npu` 内部属性**：`get_cann_version` 读取 `torch_npu.version` / `torch_npu.cann_version`，不同 `torch_npu` 版本属性名可能差异，需 fallback 多个属性名。
- **NPU autocast 日志仅首次打印**：若 CANN 版本在训练中途变化（极端场景），日志不会更新；但实际场景 CANN 版本在进程生命周期内不变，可接受。
- **ROCm 版本探测依赖 `torch.version.hip`**：非 ROCm build 的 PyTorch 此属性为 `None`，`has_rocm` 返回 `False`；用户安装 NVIDIA CUDA build 的 PyTorch 时 `has_rocm` 永远 `False`，符合预期但需文档说明。

### 风险与缓解

| 风险 | 缓解策略 |
|---|---|
| 用户在 NVIDIA GPU 环境误用 `--device rocm` | `_parse_device("rocm")` 识别字符串，`get_backend("rocm:0")` 返回 `TorchBackend(device="cuda:0")` 内部走 cuda 路径；`has_rocm()` 返回 `False` 时打印警告"当前 PyTorch 非 ROCm build，--device rocm 将走 cuda 路径"，用户得到明确信号 |
| `torch_npu` 版本升级破坏 CANN 版本探测 | `get_cann_version` 多属性 fallback（`cann_version` / `version` / `__cann_version__`）；测试 `test_rocm_npu_compat.py` 在无 NPU 环境跳过相关断言 |
| `autocast` NPU 日志在多进程训练重复打印 | `_npu_autocast_logged` 是模块级全局，每个进程仅打印一次；DDP 多 rank 场景各 rank 各打印一次（可接受，便于诊断各 rank 环境） |
| `get_memory_info("npu")` 多 API fallback 仍失败 | 兜底返回 `{"allocated": 0, "total": 0}` + 警告日志，不抛异常（避免显存查询失败阻塞训练） |
| ROCm build 的 PyTorch 与 NVIDIA CUDA build 行为差异 | ROCm build 通过 HIP 暴露 cuda API，`torch.cuda.is_available()` 返回 `True`，`torch.cuda.device_count()` 返回 AMD GPU 数量；测试覆盖 ROCm 环境（无 ROCm 环境跳过相关断言） |
| `--device rocm` 在无 ROCm 环境静默失败 | `spark/run.py` 启动时打印设备信息（type + 版本），`has_rocm() == False` 时显式警告；用户启动即可发现环境问题 |

## 替代方案（已否决）

### 方案 A：自研 ROCm kernel（HIP / PyCUDA HIP）

**描述**：用 HIP C++ 或 PyCUDA HIP 直接写 ROCm kernel 实现 matmul / attention。

**否决理由**：
- 维护成本极高（每种 op 都要写 kernel + 反向 kernel，且 AMD GPU 架构迭代频繁）
- 性能不及 PyTorch 原生（PyTorch ROCm build 已集成 MIOpen / rocBLAS / FlashAttention-ROCm，经过 AMD 工程团队大规模优化）
- 违反 ADR-001 "不重新发明轮子"原则
- NPU 无法复用（HIP 仅支持 AMD GPU）

### 方案 B：自研 CANN kernel（Ascend C++ / MindSpore IR）

**描述**：用 Ascend C++ 或 MindSpore IR 直接写 NPU kernel 实现 VerseNex 算子。

**否决理由**：
- 维护成本极高（每种 op 都要写 kernel + 反向 kernel，且 Ascend 架构迭代频繁）
- 性能不及 `torch_npu` 原生（`torch_npu` 已集成 HCCL / hccl_kernel，经过华为工程团队大规模优化）
- 违反 ADR-001 "不重新发明轮子"原则
- 与 ADR-005 "NPU 走 `torch_npu`"决策冲突

### 方案 C：强制 HIP（删除 cuda 路径）

**描述**：仅支持 ROCm / HIP，删除 NVIDIA CUDA 路径。

**否决理由**：
- 破坏 NVIDIA GPU 用户生态（NVIDIA CUDA 仍是 GPU 训练主流）
- 违反 ADR-005 "委托 PyTorch 多后端"决策
- PyTorch ROCm build 与 NVIDIA CUDA build 共享 `cuda` API（HIP-on-ROCm），无需二选一

### 方案 D：仅支持 `cuda`，不支持 `rocm`（保持 Part5K1 现状）

**描述**：ROCm 用户被迫用 `--device cuda`，不显式识别 `"rocm"` 字符串。

**否决理由**：
- 运维诊断困难（`device_type == "cuda"` 无法区分 NVIDIA 与 AMD，故障排查时易误判）
- 与"自研原生架构生态兼容"宗旨冲突
- 用户心智负担重（AMD GPU 用户需知晓"ROCm 走 cuda 路径"的隐式映射）

### 方案 E：用 `numba.cuda` / `cupy` 替代 PyTorch ROCm

**描述**：用 `numba.cuda` 或 `cupy` 直接写 ROCm kernel，不依赖 PyTorch ROCm build。

**否决理由**：
- `numba.cuda` 主要支持 NVIDIA CUDA，ROCm 支持不成熟
- `cupy` 对 ROCm 支持有限（`cupy.cuda` 走 HIP，但 API 不完整）
- 违反 ADR-005 "委托 PyTorch"决策
- 失去 PyTorch autograd 委托的开销优势

### 方案 F：用 `torch_npu` 自研 NPU 算子（绕过 HCCL）

**描述**：用 `torch_npu` 的底层 API 自研 NPU 算子，绕过 HCCL / hccl_kernel。

**否决理由**：
- 维护成本极高（每种 op 都要写 NPU 算子 + 反向算子）
- 性能不及 `torch_npu` 原生算子（HCCL / hccl_kernel 已经过华为工程团队优化）
- 违反 ADR-005 "NPU 走 `torch_npu`"决策
- 与"不自研 kernel"约束冲突

## 备注

- 本 ADR 是 ADR-005 "GPU/NPU 后端抽象"在 ROCm / CANN 生态兼容场景的具体落地，不否定 ADR-005 的"委托 PyTorch / 不自研 kernel"原则
- `torch` 与 `torch_npu` 均为**可选依赖**，`pip install verse-torch` 不会拉取它们；`has_rocm` / `has_cann` 在无 PyTorch / `torch_npu` 环境返回 `False`，不抛异常
- ROCm 路径**复用 cuda API**（HIP-on-ROCm），不自研 ROCm kernel；NPU 路径**复用 `torch_npu`**，不自研 CANN kernel
- 相关测试：`tests/test_rocm_npu_compat.py` 覆盖 ROCm/CANN 探测 API + device 字符串识别 + autocast 等价性（无 ROCm/NPU 环境时跳过相关断言）；现有 `tests/test_device_backend.py` 零回归（CPU 路径不变）
- 相关代码：
  - [`verse_torch/device.py`](../../packages/verse_torch/verse_torch/device.py) —— `_parse_device` 支持 `"rocm"` + `has_rocm` / `has_cann` / `get_rocm_version` / `get_cann_version` + `get_memory_info("npu")` 多 API 兜底
  - [`verse_torch/backend_torch.py`](../../packages/verse_torch/verse_torch/backend_torch.py) —— `TorchBackend` 兼容 `"rocm"` device + `autocast` ROCm → cuda 映射 + NPU 首次启用日志
  - [`spark/run.py`](../../spark/run.py) —— `--device rocm` / `--device rocm:0` 识别 + 启动设备信息打印（`_print_device_info`）

## 演进路线

- **ROCm FlashAttention-ROCm 适配**：若未来 VerseNex 的 `TriSparseAttention` 需要在 ROCm 上启用 FlashAttention-ROCm 加速，可在 `backend_torch.py` 增加 ROCm 路径的 attention 算子委托（仍走 PyTorch `scaled_dot_product_attention`，不自研 kernel）
- **CANN 多 NPU 卡训练**：若未来需要支持多 NPU 卡分布式训练（HCCL），可在 `device.py` 增加 `npu:0` / `npu:1` 多卡识别 + `get_memory_info("npu:0")` 单卡查询
- **Intel XPU 兼容**：若未来需要支持 Intel XPU（`torch_xpu` / IPEX），可参考本 ADR 模式新增 `has_xpu` / `get_xpu_version` + `_parse_device("xpu")` + autocast 映射
