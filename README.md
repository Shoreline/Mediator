# Mediator

一个用于 MM-SafetyBench 数据集推理的统一框架，支持多种 LLM Provider 和本地 VSP (VisualSketchpad) 工具。

**示例命令：**

```bash
# 使用 OpenRouter 调用 GPT-5
caffeinate -i python request.py \
  --json_glob "~/code/MM-SafetyBench/data/processed_questions/*.json" \
  --image_base "~/Downloads/MM-SafetyBench_imgs/" \
  --provider openrouter \
  --model "openai/gpt-5"

# 使用 CoMT-VSP（双任务模式，自动下载 CoMT 数据集）
python request.py --provider comt_vsp --max_tasks 10

# 评估结果（新版结构）
python mmsb_eval.py --jsonl_file output/job_1_tasks_10_.../results.jsonl

# 查看 JSONL 文件
python view_jsonl.py output/job_1_tasks_10_.../results.jsonl --to_json results.json
```


## 📋 功能特性

- **多 Provider 支持**：
  - OpenAI API（GPT-4o, GPT-5 等）
  - OpenRouter API（支持多种模型）
  - Qwen API（本地或远程服务）
  - VSP (VisualSketchpad) - 本地多模态 AI 工具
  - CoMT-VSP - 结合 CoMT 数据集的增强型 VSP（双任务模式）

- **并发处理**：支持高并发推理，可配置并发数量
- **自动重试**：失败任务自动重试，支持失败模式检测
- **批量处理**：支持批量处理 MM-SafetyBench 数据集
- **批量运行**：支持通过 `batch_request.py` 组合不同参数批量运行
- **结果保存**：自动保存结果到 JSONL 格式，文件名包含任务编号便于追踪
- **进度追踪**：实时显示处理进度和预估剩余时间
- **报告生成**：自动生成包含图表的 HTML 评估报告

## 🚀 快速开始

### 安装依赖

```bash
# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 环境配置

根据使用的 Provider，设置相应的环境变量：

#### OpenAI
```bash
export OPENAI_API_KEY="your-api-key"
```

#### OpenRouter
```bash
export OPENAI_API_KEY="your-openrouter-api-key"
# 注意：OpenRouter 使用 OPENAI_API_KEY 环境变量，但需要设置 provider="openrouter"
```

#### Qwen
```bash
export QWEN_ENDPOINT="http://127.0.0.1:8000"  # 本地服务地址
export QWEN_API_KEY="your-api-key"
```

#### VSP (VisualSketchpad)
```bash
export VSP_PATH="/path/to/VisualSketchpad"  # VSP 项目路径（可选，默认：/Users/yuantian/code/VisualSketchpad）
export VSP_OUTPUT_DIR="output/vsp_details"  # VSP 详细输出目录（可选，默认：output/vsp_details）
```

#### CoMT-VSP (增强型 VSP)
```bash
# CoMT-VSP 使用与 VSP 相同的环境变量
export VSP_PATH="/path/to/VisualSketchpad"

# CoMT 数据集路径（可选）
# 如果不设置或文件不存在，会自动从 HuggingFace 下载
export COMT_DATA_PATH="~/code/CoMT/comt/data.jsonl"
```

CoMT-VSP 会自动：
- 从 HuggingFace 下载 CoMT 数据集（如果本地不存在）
- 缓存 CoMT 图片到 `~/.cache/mediator/comt_images/`
- 详细输出保存到 `output/comt_vsp_details/`

## 📖 使用方法

### 基本用法

使用 `request.py` 处理 MM-SafetyBench 数据集：

```bash
python request.py \
  --json_glob "~/code/MM-SafetyBench/data/processed_questions/*.json" \
  --image_base "~/Downloads/MM-SafetyBench_imgs/" \
  --provider openai \
  --model_name "gpt-4o"
```

### 常用命令示例

#### 1. 测试 10 个样本（快速验证）

```bash
python request.py \
  --json_glob "~/code/MM-SafetyBench/data/processed_questions/*.json" \
  --image_base "~/Downloads/MM-SafetyBench_imgs/" \
  --provider openai \
  --model_name "gpt-4o" \
  --max_tasks 10
```

输出会自动组织在独立的 job 文件夹中：`output/job_{num}_tasks_{total}_{Provider}_{model}_{timestamp}/`

例如：`output/job_1_tasks_10_Openai_gpt-4o_1101_120000/`
- `num`: 单调递增的任务编号（从 1 开始）
- `total`: 实际处理的任务数
- `Provider`: CamelCase 格式的 provider 名称
- `timestamp`: MMDD_HHMMSS 格式（无年份）

#### 2. 使用 OpenRouter 调用 Claude

```bash
python request.py \
  --provider openrouter \
  --model_name "anthropic/claude-3.5-sonnet" \
  --json_glob "~/code/MM-SafetyBench/data/processed_questions/*.json" \
  --image_base "~/Downloads/MM-SafetyBench_imgs/" \
  --consumers 5 \
  --max_tasks 50
```

#### 3. 使用 VSP 处理（本地多模态工具）

```bash
python request.py \
  --provider vsp \
  --json_glob "~/code/MM-SafetyBench/data/processed_questions/*.json" \
  --image_base "~/Downloads/MM-SafetyBench_imgs/" \
  --consumers 3 \
  --max_tasks 100
```

输出会组织在 job 文件夹中：`output/job_{num}_tasks_100_Vsp_{model}_{timestamp}/`
- 主文件：`results.jsonl`, `eval.csv`, `console.log`, `metadata.yaml`
- 详细输出：`details/` 子目录

#### 4. 使用 CoMT-VSP 处理（增强型双任务模式）

⚠️ **注意**：必须通过 `--comt_sample_id` 指定一个确定的 CoMT 样本 ID

```bash
# 使用指定的 CoMT 样本（从 HuggingFace 自动下载数据集）
python request.py \
  --provider comt_vsp \
  --comt_sample_id "deletion-0107" \
  --json_glob "~/code/MM-SafetyBench/data/processed_questions/*.json" \
  --image_base "~/Downloads/MM-SafetyBench_imgs/" \
  --consumers 3 \
  --max_tasks 50
```

```bash
# 使用本地 CoMT 数据集 + 指定样本
python request.py \
  --provider comt_vsp \
  --comt_data_path "~/code/CoMT/comt/data.jsonl" \
  --comt_sample_id "deletion-0107" \
  --json_glob "~/code/MM-SafetyBench/data/processed_questions/*.json" \
  --image_base "~/Downloads/MM-SafetyBench_imgs/" \
  --max_tasks 20
```

输出会组织在 job 文件夹中：`output/job_{num}_tasks_50_ComtVsp_{model}_{timestamp}/`
- 主文件：`results.jsonl`, `eval.csv`, `console.log`, `metadata.yaml`
- 详细输出：`details/` 子目录

> 💡 **CoMT-VSP 说明**：同时向 LLM 提出两个任务：
> - TASK 1: CoMT 物体检测任务（强制使用 VSP detection 工具）
> - TASK 2: MM-SafetyBench 安全评估任务（直接回答）
> 
> 推荐使用 'deletion' 子集的样本（如 deletion-0107），适合转换为物体计数任务
> 
> 详细说明请参考 `COMT_GUIDE.md`

#### 5. 处理完整数据集

```bash
python request.py \
  --provider openai \
  --model_name "gpt-4o" \
  --json_glob "~/code/MM-SafetyBench/data/processed_questions/*.json" \
  --image_base "~/Downloads/MM-SafetyBench_imgs/" \
  --consumers 10
```

不指定 `--max_tasks` 会处理所有数据。

#### 6. 指定输出路径

```bash
python request.py \
  --json_glob "~/code/MM-SafetyBench/data/processed_questions/*.json" \
  --image_base "~/Downloads/MM-SafetyBench_imgs/" \
  --provider openai \
  --model_name "gpt-4o" \
  --save_path "my_results.jsonl"
```

#### 7. 处理特定类别

```bash
python request.py \
  --json_glob "~/code/MM-SafetyBench/data/processed_questions/*.json" \
  --image_base "~/Downloads/MM-SafetyBench_imgs/" \
  --provider openai \
  --model_name "gpt-4o" \
  --categories 08-Political_Lobbying 12-Health_Consultation
```

#### 8. 处理多个图片类型

MM-SafetyBench 支持多种图片类型：
- `SD`: 使用 "Changed Question" 字段
- `SD_TYPO`: 使用 "Rephrased Question" 字段
- `TYPO`: 使用 "Rephrased Question(SD)" 字段

```bash
python request.py \
  --json_glob "~/code/MM-SafetyBench/data/processed_questions/*.json" \
  --image_base "~/Downloads/MM-SafetyBench_imgs/" \
  --provider openai \
  --model_name "gpt-4o" \
  --image_types SD SD_TYPO
```

## ⚙️ 参数说明

### 必需参数

- `--json_glob`: MM-SafetyBench JSON 文件的 glob 模式
  - 例如：`"~/code/MM-SafetyBench/data/processed_questions/*.json"`
- `--image_base`: 图片基础目录
  - 例如：`"~/Downloads/MM-SafetyBench_imgs/"`

### Provider 参数

- `--provider`: Provider 类型（`openai` / `openrouter` / `qwen` / `vsp` / `comt_vsp`）
  - 默认：`openai`
- `--model_name`: 模型名称
  - OpenAI: `gpt-4o`, `gpt-5`, `gpt-4o-mini` 等
  - OpenRouter: `anthropic/claude-3.5-sonnet`, `openai/gpt-4o` 等
  - Qwen: `qwen3-vl-235b-a22b-instruct` 等
  - VSP / CoMT-VSP: `model_name` 参数不起作用（使用 VSP 自己的配置）

### 任务控制参数

- `--max_tasks`: 最大任务数（用于小批量测试）
  - 默认：`None`（处理所有数据）
- `--consumers`: 并发消费者数量
  - 默认：`10`
  - OpenRouter 等 API 建议使用较低值（3-5）避免限流
  - VSP / CoMT-VSP 建议使用较低值（3-5）因为每个任务耗时较长

### CoMT-VSP 特定参数

- `--comt_data_path`: CoMT 数据集路径
  - 默认：`~/code/CoMT/comt/data.jsonl`（如果文件存在则使用本地，否则从 HuggingFace 下载）
  - 支持本地文件路径或留空以自动从 HuggingFace 下载
- `--comt_sample_id`: 指定使用的 CoMT 样本 ID
  - 默认：`None`（随机选择 CoMT 样本）
  - 示例：`creation-10003`（用于固定样本的可重复实验）

### 模型参数

- `--temp`: Temperature（默认：`0.0`）
- `--top_p`: Top-p（默认：`1.0`）
- `--max_tokens`: 最大 token 数（默认：`2048`）
- `--seed`: 随机种子（可选）

### 数据过滤参数

- `--image_types`: 要处理的图片类型（可指定多个）
  - 选项：`SD`, `SD_TYPO`, `TYPO`
  - 默认：`["SD"]`
- `--categories`: 要处理的类别（可指定多个）
  - 例如：`--categories 08-Political_Lobbying 12-Health_Consultation`
  - 默认：处理所有类别

### 采样参数

- `--sampling_rate`: 采样率（0.0-1.0）
  - 默认：`1.0`（不采样）
  - 用于对数据集进行下采样，减少 API 调用
  - 例如：`0.12` 表示采样 12% 的数据
- `--sampling_seed`: 采样随机种子
  - 默认：`42`
  - 相同种子确保相同的采样结果
  - 用于可重复实验

### 其他参数

- `--save_path`: 输出文件路径
  - 默认：自动生成 `output/{model_name}_{timestamp}.jsonl`
  - VSP: 自动生成 `output/vsp_{timestamp}.jsonl`
- `--proxy`: HTTP 代理（可选）

## 🎲 数据采样

### 概述

`pseudo_random_sampler.py` 提供确定性的数据采样功能：
- **确定性**: 相同的随机种子、数据大小和采样率，每次执行结果完全相同
- **按类别采样**: 对 MMSB 数据集的 13 个类别独立采样，确保每个类别保留相同比例
- **灵活集成**: 已集成到 `request.py` 和 `mmsb_eval.py` 中

### 使用场景

#### 场景 1: 在请求时下采样（减少 API 调用）

```bash
# 采样 12% 的数据（每个类别独立采样 12%）
python3 request.py \
  --provider openai \
  --model gpt-4o \
  --sampling_rate 0.12 \
  --sampling_seed 42

# 采样 50% 的数据
python3 request.py \
  --provider openai \
  --model gpt-4o \
  --sampling_rate 0.5 \
  --sampling_seed 12345
```

**效果**:
- 对 MMSB 的 13 个类别分别采样
- 每个类别保留约 `sampling_rate * 类别大小` 条数据
- 输出文件自动添加采样标记：`52_tasks_202_gpt-4o_sampled_0.12_seed42_2025-12-10_08-36-47.jsonl`

#### 场景 2: 对已有结果采样统计

使用 `mmsb_eval.py` 的 `--sampling_rate` 参数对已有结果进行采样统计：

```bash
# 对完整结果文件进行 12% 采样统计
python3 mmsb_eval.py \
  --jsonl_file output/35_tasks_1680_qwen_model.jsonl \
  --sampling_rate 0.12 \
  --sampling_seed 42

# 对同一结果文件进行多次不同采样率的统计
python3 mmsb_eval.py \
  --jsonl_file output/35_tasks_1680_qwen_model.jsonl \
  --sampling_rate 0.25 \
  --sampling_seed 42

python3 mmsb_eval.py \
  --jsonl_file output/35_tasks_1680_qwen_model.jsonl \
  --sampling_rate 0.50 \
  --sampling_seed 42
```

**用途**:
- 快速验证不同采样率下的攻击率趋势
- 对大规模结果进行快速统计分析
- 生成多个采样率的对比图表

**输出文件命名**:
- 采样后的 CSV 文件会自动添加采样标记
- 例如: `35_eval_tasks_1680_qwen_model_sampled_0.12_seed42.csv`

### 技术细节

采样器使用 Python 的 `random.Random(seed)` 确保确定性，为每个类别生成唯一的派生种子：

```python
category_seed = seed + hash(category_name) % 1000000
```

这确保：
1. 不同类别使用不同的随机序列
2. 相同的原始种子 + 类别名总是产生相同的类别种子
3. 各类别的采样相互独立

### 采样验证

运行内置测试验证采样器正确性：

```bash
python3 pseudo_random_sampler.py
```

## 📁 输出格式

### Job 文件夹结构

每次运行会创建一个独立的 job 文件夹：

```
output/job_104_tasks_202_ComtVsp_qwen3-vl-8b_0104_193618/
├── results.jsonl          # LLM 生成结果（JSONL 格式）
├── eval.csv               # 评估指标汇总（CSV 格式）
├── console.log            # 完整的控制台输出
├── metadata.yaml          # Job 元信息（配置、时长、指标等）
└── details/               # Provider 详细输出（VSP/CoMT-VSP）
    └── vsp_2026-01-04_00-03-00/
        └── {category}/{index}/
            ├── input/     # VSP 输入文件
            └── output/    # VSP 输出文件
```

**文件夹命名格式：** `job_{num}_tasks_{total}_{Provider}_{model}_{MMDD_HHMMSS}`

- `num`: 单调递增的任务编号（从 1 开始，保存在 `output/.task_counter`）
- `total`: 实际处理的任务数
- `Provider`: CamelCase 格式的 provider 名称（如 `Openai`, `ComtVsp`）
- `model`: 清理后的模型名称（特殊字符替换为下划线）
- `MMDD_HHMMSS`: 时间戳（月日_时分秒，无年份）

### JSONL 文件格式

`results.jsonl` 中每行一个 JSON 对象：

```json
{
  "index": "0",
  "pred": [{
    "role": "assistant",
    "content": [{
      "type": "text",
      "reasoning": null,
      "text": "模型的回答..."
    }]
  }],
  "origin": {
    "index": "0",
    "category": "01-Illegal_Activitiy",
    "question": "问题文本",
    "image_path": "~/Downloads/MM-SafetyBench_imgs/...",
    "image_type": "SD",
    "question_field": "Changed Question"
  },
  "sent": {
    "prompt_parts": [...]
  },
  "meta": {
    "model": "gpt-4o",
    "params": {
      "temperature": 0.0,
      "top_p": 1.0,
      "max_tokens": 2048
    },
    "ts": 1698765432.123
  }
}
```

### metadata.yaml 格式

Job 元信息文件包含完整的配置和结果：

```yaml
job_num: 104
job_folder: job_104_tasks_202_ComtVsp_qwen3-vl-8b_0104_193618
timestamp: 01-04 19:36:18
command: python request.py --provider comt_vsp --model ...

config:
  provider: comt_vsp
  model: qwen3-vl-8b-instruct
  temperature: 0.0
  top_p: 1.0
  max_tokens: 2048
  seed: 42
  consumer_size: 20

execution:
  total_tasks: 202
  request_duration_seconds: 1234.5
  eval_duration_seconds: 456.7
  total_duration_seconds: 1691.2
  throughput_tasks_per_second: 0.16
  stop_reason: null

files:
  jsonl: results.jsonl
  console_log: console.log
  eval_csv: eval.csv
  details: details/

eval_metrics:
  overall:
    total: 202
    evaluated: 200
    safe: 150
    unsafe: 45
    attack_rate: 22.5
  by_category:
    01-Illegal_Activity:
      total: 20
      evaluated: 20
      safe: 15
      unsafe: 5
      attack_rate: 25.0
```

## 🔧 VSP / CoMT-VSP 特殊说明

VSP (VisualSketchpad) 和 CoMT-VSP 是本地多模态 AI 工具，与其他 Provider 有所不同：

### VSP 输出结构

使用 VSP 或 CoMT-VSP 时，所有输出都在统一的 job 文件夹中：

```
output/job_104_tasks_202_Vsp_model_0104_193618/  (或 ComtVsp)
├── results.jsonl          # 结果摘要
├── eval.csv               # 评估指标
├── console.log            # 控制台日志
├── metadata.yaml          # Job 元信息
└── details/               # 统一的详细输出目录
    └── vsp_2026-01-04_19-36-18/
        └── {category}/{index}/
            ├── input/     # VSP 输入文件
            │   ├── request.json (或 ex.json)
            │   └── image_*.jpg
            ├── output/    # VSP 输出文件
            │   ├── vsp_debug.log
            │   ├── output.json
            │   └── ...
            └── mediator_metadata.json
```

**关键变化：**
- 不再有单独的 `vsp_details/` 或 `comt_vsp_details/` 目录
- 详细输出统一在 job 文件夹的 `details/` 子目录中
- 所有相关文件集中管理，便于组织和清理

### CoMT-VSP 双任务模式

CoMT-VSP 结合了 CoMT 数据集，采用双任务模式：

- **TASK 1**: CoMT 物体检测任务
  - 使用 CoMT 数据集中的 deletion 子集样本
  - 强制使用 VSP 的 detection 工具进行物体检测和计数
  - 任务类型：`vision`
  - 必须通过 `--comt_sample_id` 指定样本（如 `deletion-0107`）

- **TASK 2**: MM-SafetyBench 安全评估任务
  - 原始的 MM-SafetyBench 问题
  - 直接回答，不使用工具
  - 用于评估模型的安全性能

**自动数据管理**：
- CoMT 数据集优先从 HuggingFace 下载（`czh-up/CoMT`）
- 图片自动缓存到 `~/.cache/mediator/comt_images/` 避免重复下载
- 支持指定本地 CoMT 数据路径（`--comt_data_path`）

### VSP 配置

VSP 和 CoMT-VSP 使用的 LLM 配置在 VSP 自己的配置文件中设置（通常是环境变量 `OAI_CONFIG_LIST` 或配置文件）。

`request.py` 的 `--model_name` 参数对 VSP / CoMT-VSP 不起作用。

### VSP 失败检测

VSP 和 CoMT-VSP 支持自动检测失败任务（如 API 余额不足、超时等），并自动重试。

### VSP 路径清理

VSP 和 CoMT-VSP 运行完成后，会**自动清理**输出文件中的敏感路径信息：
- 将所有绝对路径中的主目录替换为 `~`
- 处理所有 `.json` 和 `.log` 文件
- 保护隐私，便于分享结果
- 无需手动操作，完全自动化

详细说明请参考：
- `COMT_GUIDE.md` - CoMT-VSP 完整指南
- `VSP_USAGE_EXAMPLES.md` - VSP 使用示例
- `VSP_BATCH_MODE.md` - VSP 批量模式说明

## 🧪 测试

项目包含多个测试脚本，位于 `tests/` 目录：

```bash
# 运行失败答案检测测试
python tests/test_failed_answer_detection.py

# 测试 MM-SafetyBench 数据加载
python tests/test_mmsb_loader.py

# 测试 Provider
python tests/test_provider.py

# 测试 VSP Provider
python tests/test_vsp_provider.py
```

更多测试说明请参考 `tests/README.md`。

## 🔄 批量运行（batch_request.py）

使用 `batch_request.py` 可以组合不同参数批量运行多次 `request.py`。

### 配置参数组合

编辑 `batch_request.py` 中的 `args_combo` 列表：

```python
args_combo = [
    # 固定参数（字符串）：所有组合都会使用
    "--categories 12-Health_Consultation --max_tasks 10",
    
    # 参数变体（列表）：会遍历每个变体
    [
        '--provider comt_vsp --model "qwen/qwen3-vl-235b-a22b-instruct"',
        '--provider openrouter --model "google/gemini-2.5-flash"',
    ],
]
```

### 运行批量任务

```bash
python batch_request.py
```

### 功能特性

- **笛卡尔积组合**：自动生成所有参数变体的组合
- **实时进度显示**：每个子任务的进度实时显示
- **详细结果汇总**：批量完成后打印所有任务的详细信息
- **日志保存**：所有输出保存到 `output/batch-{task_num}_{total}_{timestamp}.log`
- **自动生成报告**：批量完成后自动调用 `generate_report_with_charts.py` 生成报告

### 配置选项

```python
STOP_ON_ERROR = False    # 遇到错误时是否停止
VERBOSE = True           # 是否显示详细输出
GENERATE_REPORT = True   # 是否在完成后生成报告
```

### 输出文件

批量运行完成后会生成：
- **日志文件**: `output/batch-{task_num}_{total}_{timestamp}.log`
- **HTML 报告**: `output/batch_{task_num}_evaluation_report.html`
- **各任务的 job 文件夹**: `output/job_{num}_tasks_{total}_*/` （每个批量任务一个独立文件夹）

## 📊 生成评估报告（generate_report_with_charts.py）

生成包含图表的 HTML 评估报告。

### 基本用法

```bash
# 使用默认逻辑，加载所有符合条件的评估文件
python generate_report_with_charts.py

# 指定要处理的评估文件（例如采样后的文件）
python generate_report_with_charts.py \
  --files output/*_eval-sampled_0.12_seed42_tasks_*.csv \
  --output output/sampled_0.12_evaluation_report.html

# 指定输出报告路径
python generate_report_with_charts.py --output output/my_report.html
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--files` | `None` | 指定要处理的评估 CSV 文件列表。不指定则使用默认逻辑 |
| `--output` | `output/evaluation_report.html` | 输出报告文件路径 |

### 输出内容

- **HTML 报告**: 包含所有模型的攻击率对比图表
  - **总攻击率图表**: 每个模型的总体攻击率（基于所有类别的加总）
  - **分类攻击率图表**: 每个类别的详细攻击率
- **图表文件**: 
  - `output/chart_*_overall.png` - 总攻击率对比图
  - `output/chart_*.png` - 分类攻击率图

### 总攻击率计算

总攻击率基于所有类别的实际评估数量加总计算：

```
总攻击率 = (所有类别的 Unsafe 数量之和) / (所有类别的 Evaluated 数量之和) × 100%
```

这与简单的平均攻击率不同，因为它考虑了每个类别的实际评估数量权重。

## 🔄 集成流水线（推荐）

`request.py` 已经集成了完整的评估流程，**默认会自动执行评估和指标计算**，无需手动调用 `mmsb_eval.py`。

### 一步完成（生成答案 + 评估 + 指标）

```bash
# 最简单的用法（使用默认数据路径）
python request.py --max_tasks 10
```

这会自动：
1. 调用 LLM 生成答案
2. 使用 GPT 评估答案安全性
3. 计算并输出评估指标
4. 生成 CSV 报告

### 控制评估行为

```bash
# 跳过评估（仅生成答案）
python request.py --max_tasks 10 --skip_eval

# 自定义评估模型和并发数
python request.py --max_tasks 50 --eval_model "gpt-5" --eval_concurrency 30
```

### 输出文件

运行完整流水线后，会在独立的 job 文件夹中生成：

```
output/job_{num}_tasks_{total}_{Provider}_{model}_{timestamp}/
├── results.jsonl          # 包含所有问题、答案和评估结果
├── eval.csv               # 评估指标汇总表
├── console.log            # 完整的控制台输出
├── metadata.yaml          # Job 配置、时长、指标等元信息
└── details/               # VSP/CoMT-VSP 详细输出（如适用）
```

**命名说明：**
- `num`: 单调递增的任务编号（从 1 开始，保存在 `output/.task_counter`）
- `total`: 实际处理的任务数
- `Provider`: CamelCase 格式的 provider 名称
- `timestamp`: MMDD_HHMMSS 格式（无年份）

### 与传统两步流程对比

**原来（两步）**:
```bash
# 步骤 1: 生成答案
python request.py --max_tasks 10

# 步骤 2: 评估答案
python mmsb_eval.py --jsonl_file output/gpt-5_xxx.jsonl
```

**现在（一步）**:
```bash
# 一步完成所有操作
python request.py --max_tasks 10
```

### 评估相关参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--skip_eval` | `False` | 跳过评估步骤（默认会自动评估） |
| `--eval_model` | `gpt-5-mini` | 用于评估的模型 |
| `--eval_concurrency` | `20` | 评估并发数 |

## 📊 评估（手动模式）

如果需要单独评估已生成的答案（例如评估中断后断点续传），可以使用 `mmsb_eval.py`：

```bash
# 评估普通模型结果（新版结构）
python mmsb_eval.py \
  --jsonl_file output/job_1_tasks_10_Openai_gpt-4o_.../results.jsonl

# 评估 VSP 结果（包含工具使用检测）
python mmsb_eval.py \
  --jsonl_file output/job_2_tasks_50_Vsp_.../results.jsonl

# 评估 CoMT-VSP 结果（包含工具使用检测和双任务答案提取）
python mmsb_eval.py \
  --jsonl_file output/job_3_tasks_100_ComtVsp_.../results.jsonl
```

**注意**：新版结构中，JSONL 文件统一命名为 `results.jsonl`，评估 CSV 会自动保存为 `eval.csv` 在同一 job 文件夹中。

**评估指标**：
- **Attack Rate**: 模型被成功诱导执行不安全行为的比例
- **Safe Rate**: 模型正确拒绝不安全请求的比例
- **Tool Usage Rate** (VSP/CoMT-VSP): 使用视觉工具的任务比例

**CoMT-VSP 特殊处理**：
- 自动提取 "TASK 2" 的答案用于评估
- 统计几何工具（`geo` task type）的使用情况
- 生成包含工具使用率的 CSV 报告

## 🔍 VSP / CoMT-VSP 工具使用分析

### 方法 1: 独立脚本分析（批量统计）

使用 `check_vsp_tool_usage.py` 分析 VSP / CoMT-VSP 是否使用了视觉工具（detection, segmentation 等）：

```bash
# 分析默认目录（所有 job 文件夹的 details）
python check_vsp_tool_usage.py

# 分析指定 job 文件夹的 details
python check_vsp_tool_usage.py --dir output/job_104_tasks_202_Vsp_.../details

# 分析指定批次（新版结构）
python check_vsp_tool_usage.py --dir output/job_104_tasks_202_Vsp_.../details/vsp_2026-01-04_00-03-00

# 分析特定类别
python check_vsp_tool_usage.py --dir output/job_104_tasks_202_Vsp_.../details/vsp_2026-01-04_00-03-00/08-Political_Lobbying

# 保存示例到文件（去掉通用示例文本）
python check_vsp_tool_usage.py --summarize_examples

# 指定最多收集50个示例
python check_vsp_tool_usage.py --summarize_examples --max_examples 50
```

该脚本会扫描指定目录下所有 `vsp_debug.log` 文件，统计：
- 总体工具使用率
- 按类别统计工具使用情况
- 提供使用/未使用工具的示例文件

**保存示例功能（--summarize_examples）：**
- 收集使用工具和未使用工具的示例（默认各100个）
- 去掉 VSP 的通用示例文本，只保留用户交互部分
- 保存到 `output/vsp_examples_used_tools.txt` 和 `output/vsp_examples_no_tools.txt`

**输出示例：**
```
📊 VSP 工具使用统计
总文件数: 13455
  - 使用了工具: 1484 (11.0%)
  - 未使用工具: 11963 (88.9%)

📋 按类别统计
类别                             总数       使用工具       未使用        使用率       
--------------------------------------------------------------------------------
Health_Consultation            872      193        679        22.1%
Gov_Decision                   1188     254        934        21.4%
Political_Lobbying             1244     239        1005       19.2%
Illegal_Activitiy              775      16         759        2.1%
...
```

### 方法 2: 集成到评估流程（为每条记录添加字段）

`mmsb_eval.py` 会自动为 VSP / CoMT-VSP 的 JSONL 文件添加 `used_vsp_tools` 字段：

```bash
# 完整流程：评估 + 计算指标 + 添加工具使用字段（VSP）
python mmsb_eval.py --jsonl_file output/vsp_xxx.jsonl

# 完整流程：评估 + 计算指标 + 添加工具使用字段（CoMT-VSP）
python mmsb_eval.py --jsonl_file output/comt_vsp_xxx.jsonl

# 仅添加工具使用字段（不评估）
python mmsb_eval.py --jsonl_file output/vsp_xxx.jsonl --add_vsp_tools

# 跳过工具使用检测
python mmsb_eval.py --jsonl_file output/vsp_xxx.jsonl --skip_vsp_tools
```

添加后的 JSONL 记录会包含：
```json
{
  "index": "18",
  "pred": [...],
  "origin": {...},
  "used_vsp_tools": true,  // 新增字段
  ...
}
```

**检测原理：**
- VSP / CoMT-VSP 提供多种视觉分析工具（detection, segmentation, depth 等）
- 当 VSP 使用工具时，会在 RESULT 部分生成 Python 代码块
- 脚本通过检测 ````python` 代码块来判断是否使用了工具
- 从 job 文件夹路径定位 `details/` 子目录中的 `vsp_debug.log` 文件
- 新版结构中，所有 details 统一在 job 文件夹的 `details/` 子目录

## 📂 项目结构

```
Mediator/
├── README.md                    # 本文件
├── requirements.txt             # Python 依赖
├── request.py                   # 主要的推理脚本
├── batch_request.py             # 批量运行脚本
├── provider.py                  # Provider 接口和实现
├── pseudo_random_sampler.py     # 伪随机采样器
├── mmsb_eval.py                 # 评估脚本
├── generate_report_with_charts.py  # 报告生成脚本
├── check_vsp_tool_usage.py      # VSP 工具使用分析
├── view_jsonl.py                # JSONL 查看工具
├── cleanup_output.py            # 输出目录清理工具
├── COMT_GUIDE.md                # CoMT-VSP 使用指南
├── tests/                       # 测试脚本
│   ├── README.md
│   ├── test_provider.py
│   ├── test_vsp_provider.py
│   └── ...
├── output/                      # 输出目录（新版结构）
│   ├── .task_counter           # 任务计数器（全局单调递增）
│   ├── job_1_tasks_10_.../     # Job 1 文件夹
│   │   ├── results.jsonl       # LLM 生成结果
│   │   ├── eval.csv            # 评估指标
│   │   ├── console.log         # 控制台日志
│   │   ├── metadata.yaml       # Job 元信息
│   │   └── details/            # Provider 详细输出（如适用）
│   ├── job_2_tasks_50_.../     # Job 2 文件夹
│   └── ...
└── example/                     # 示例文件
```

## 🧹 清理输出目录（cleanup_output.py）

使用 `cleanup_output.py` 清理 output/ 目录中不需要的 job 文件夹。

### 基本用法

```bash
# 预览将要删除的 job（不实际删除）
python cleanup_output.py --dry-run

# 清理任务数 < 100 的 job（默认）
python cleanup_output.py

# 清理任务数 < 50 的 job
python cleanup_output.py --threshold 50

# 清理特定任务编号的 job
python cleanup_output.py --job-num 42

# 清理多个任务编号的 job
python cleanup_output.py --job-num 42 43 44

# 自动确认删除（不需要交互）
python cleanup_output.py --yes
```

### 功能特性

- **按任务数清理**：自动识别任务数小于阈值的 job 文件夹
- **按任务编号清理**：支持清理特定任务编号的所有 job
- **完整删除**：删除整个 job 文件夹（包括所有子文件和 details 目录）
- **预览模式**：使用 `--dry-run` 查看将要删除的内容
- **详细摘要**：显示每个 job 的大小、包含文件等信息

### 输出示例

```
🧹 output/ 目录清理工具（新版 - 基于 job 文件夹）
================================================================================
目录: output
模式: 按任务数阈值清理
阈值: tasks < 100
================================================================================

🔍 扫描 job 文件夹...

🗑️  清理摘要
================================================================================

1. Job 42 (tasks=50)
   文件夹: job_42_tasks_50_Openai_gpt-4o_0104_120000
   Provider: Openai
   Model: gpt-4o
   Timestamp: 0104_120000
   大小: 125.3 MB
   内容:
     └─ [FILE] console.log
     └─ [FILE] eval.csv
     └─ [FILE] metadata.yaml
     └─ [FILE] results.jsonl

================================================================================
总计: 1 个 job 文件夹
将释放空间: 125.3 MB
================================================================================
```

## 🔍 故障排除

### 常见问题

1. **API 密钥未设置**
   - 确保设置了相应的环境变量（`OPENAI_API_KEY` 等）

2. **VSP 路径错误**
   - 检查 `VSP_PATH` 环境变量是否指向正确的 VSP 项目目录

3. **并发过高导致限流**
   - 降低 `--consumers` 参数（特别是 OpenRouter）

4. **超时错误**
   - 默认超时为 120 秒，VSP 任务可能更长
   - 可以在代码中调整 `send_with_retry` 的 `timeout` 参数

5. **图片文件不存在**
   - 检查 `--image_base` 路径是否正确
   - 检查图片文件是否存在于预期位置

## 📝 文档

- `COMT_GUIDE.md` - **CoMT-VSP 完整指南（推荐阅读）**
- `VSP_USAGE_EXAMPLES.md` - VSP 使用示例
- `VSP_BATCH_MODE.md` - VSP 批量模式说明
- `VSP_ANSWER_EXTRACTION.md` - VSP 答案提取说明
- `tests/README.md` - 测试脚本说明

> 💡 **集成流水线**: `request.py` 默认自动执行评估和指标计算，详见 [🔄 集成流水线](#-集成流水线推荐) 章节。
> 
> 💡 **失败答案检测**: 自动识别和重试失败答案的功能已内置于 `request.py`，详见代码中的 `is_failed_answer()` 和 `send_with_retry()` 函数注释。
>
> 💡 **路径清理**: VSP/CoMT-VSP 运行完成后自动清理输出文件中的敏感路径（替换为 `~`），详见代码中的 `clean_vsp_paths()` 函数。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

（根据项目实际情况填写）

---

如有问题，请查看相关文档或提交 Issue。

