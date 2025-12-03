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

# 评估结果
python mmsb_eval.py --jsonl_file output/comt_vsp_2025-12-02_15-08-03.jsonl

# 查看 JSONL 文件
python view_jsonl.py output/comt_vsp_2025-12-02_15-08-03.jsonl --to_json results.json
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
- **结果保存**：自动保存结果到 JSONL 格式
- **进度追踪**：实时显示处理进度和预估剩余时间

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

输出文件会自动命名为：`output/gpt-4o_2025-11-01_12-00-00.jsonl`

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

输出文件：`output/vsp_2025-11-01_12-00-00.jsonl`
详细输出：`output/vsp_details/vsp_2025-11-01_12-00-00/`

#### 4. 使用 CoMT-VSP 处理（增强型双任务模式）

```bash
# 自动从 HuggingFace 下载 CoMT 数据集
python request.py \
  --provider comt_vsp \
  --json_glob "~/code/MM-SafetyBench/data/processed_questions/*.json" \
  --image_base "~/Downloads/MM-SafetyBench_imgs/" \
  --consumers 3 \
  --max_tasks 50
```

```bash
# 使用本地 CoMT 数据集
python request.py \
  --provider comt_vsp \
  --comt_data_path "~/code/CoMT/comt/data.jsonl" \
  --json_glob "~/code/MM-SafetyBench/data/processed_questions/*.json" \
  --image_base "~/Downloads/MM-SafetyBench_imgs/" \
  --max_tasks 20
```

```bash
# 使用固定的 CoMT 样本进行测试
python request.py \
  --provider comt_vsp \
  --comt_sample_id "creation-10003" \
  --max_tasks 10
```

输出文件：`output/comt_vsp_2025-11-01_12-00-00.jsonl`
详细输出：`output/comt_vsp_details/vsp_2025-11-01_12-00-00/`

> 💡 **CoMT-VSP 说明**：同时向 LLM 提出两个任务：
> - TASK 1: CoMT 几何推理任务（强制使用 VSP 几何工具）
> - TASK 2: MM-SafetyBench 安全评估任务（直接回答）
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

### 其他参数

- `--save_path`: 输出文件路径
  - 默认：自动生成 `output/{model_name}_{timestamp}.jsonl`
  - VSP: 自动生成 `output/vsp_{timestamp}.jsonl`
- `--proxy`: HTTP 代理（可选）

## 📁 输出格式

结果保存为 JSONL 格式，每行一个 JSON 对象：

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
    "image_path": "/path/to/image.jpg",
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

## 🔧 VSP / CoMT-VSP 特殊说明

VSP (VisualSketchpad) 和 CoMT-VSP 是本地多模态 AI 工具，与其他 Provider 有所不同：

### VSP 输出结构

使用 VSP 或 CoMT-VSP 时，会产生两个输出：

1. **结果摘要文件**：
   - VSP: `output/vsp_{timestamp}.jsonl`
   - CoMT-VSP: `output/comt_vsp_{timestamp}.jsonl`

2. **详细输出目录**：
   - VSP: `output/vsp_details/vsp_{timestamp}/`
   - CoMT-VSP: `output/comt_vsp_details/vsp_{timestamp}/`
   - 每个任务的完整输出
   - 目录结构：`vsp_{timestamp}/{category}/{index}/`
   - 包含：
     - `input/`: VSP 的输入文件（`request.json` / `ex.json`, `image_*.jpg`）
     - `output/`: VSP 的输出文件（`vsp_debug.log`, `output.json` 等）
     - `mediator_metadata.json`: Mediator 保存的元数据

### CoMT-VSP 双任务模式

CoMT-VSP 结合了 CoMT 数据集，采用双任务模式：

- **TASK 1**: CoMT 几何推理任务
  - 使用 CoMT 数据集中的几何问题
  - 强制使用 VSP 的几何工具（`find_perpendicular_intersection`, `find_parallel_intersection` 等）
  - 任务类型：`geo`

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

详细说明请参考：
- `COMT_GUIDE.md` - CoMT-VSP 完整指南
- `VSP_USAGE_EXAMPLES.md` - VSP 使用示例
- `VSP_BATCH_MODE.md` - VSP 批量模式说明
- `FAILED_ANSWER_DETECTION.md` - 失败答案检测功能

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

## 📊 评估

使用 `mmsb_eval.py` 对结果进行评估：

```bash
# 评估普通模型结果
python mmsb_eval.py \
  --jsonl_file output/gpt-4o_2025-11-01_12-00-00.jsonl

# 评估 VSP 结果（包含工具使用检测）
python mmsb_eval.py \
  --jsonl_file output/vsp_2025-11-01_12-00-00.jsonl

# 评估 CoMT-VSP 结果（包含工具使用检测和双任务答案提取）
python mmsb_eval.py \
  --jsonl_file output/comt_vsp_2025-12-02_15-08-03.jsonl
```

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
# 分析默认目录（所有批次）
python check_vsp_tool_usage.py

# 分析指定 VSP 批次
python check_vsp_tool_usage.py --dir output/vsp_details/vsp_2025-11-12_20-18-34

# 分析指定 CoMT-VSP 批次
python check_vsp_tool_usage.py --dir output/comt_vsp_details/vsp_2025-12-02_15-08-03

# 分析特定类别
python check_vsp_tool_usage.py --dir output/vsp_details/vsp_2025-11-12_20-18-34/08-Political_Lobbying

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
- 从 JSONL 文件名提取时间戳，定位对应的 `vsp_debug.log` 文件
- CoMT-VSP 的日志文件位于 `output/comt_vsp_details/` 目录

## 📂 项目结构

```
Mediator/
├── README.md                    # 本文件
├── requirements.txt             # Python 依赖
├── request.py                   # 主要的推理脚本
├── provider.py                  # Provider 接口和实现
├── mmsb_eval.py                 # 评估脚本
├── view_jsonl.py                # JSONL 查看工具
├── tests/                       # 测试脚本
│   ├── README.md
│   ├── test_provider.py
│   ├── test_vsp_provider.py
│   └── ...
├── output/                      # 输出目录
│   ├── *.jsonl                 # 推理结果
│   └── vsp_details/            # VSP 详细输出
└── docs/                        # 文档（如果有）
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
- `FAILED_ANSWER_DETECTION.md` - 失败答案检测功能说明
- `INTEGRATED_PIPELINE.md` - 集成流程说明
- `tests/README.md` - 测试脚本说明

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

（根据项目实际情况填写）

---

如有问题，请查看相关文档或提交 Issue。

