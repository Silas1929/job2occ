# 招聘数据云端处理 Pipeline

面向中国招聘数据的大规模职业标准化流水线，将原始招聘帖子的岗位描述依次经过 **数据清洗 → LLM 标准化 → 语义向量匹配（+可选精排） → O\*NET 映射**，最终输出按年份和职业大类分区的 Parquet 数据集。

---

## 功能介绍

### 整体目标

对 2015–2024 年共约 **1.6 亿条**招聘记录的岗位职责文本进行处理，产出统一、可分析的职业分类结果。

### 四步流水线

```
原始 CSV
     │
     ▼
Step 1: 数据清洗
     │  按 rec_id 去重 + 过滤过短描述
     │  输出: data/input/recruitment_infos_content_{year}_step1_clean.csv
     ▼
Step 2: LLM 岗位职责标准化
     │  提取并规范化职位描述中的核心职责条目
     │  输出: data/output/standardized/year={year}/data.parquet
     ▼
Step 3: 向量语义匹配（职业分类大典 2022）+ 可选精排
     │  将标准化职责映射至《中华人民共和国职业分类大典（2022）》四级编码
     │  可选启用 BGE-Reranker-v2-M3 对 top-k 候选精排
     │  输出: data/output/matched/year={year}/data.parquet
     ▼
Step 4: O*NET 国际职业编码映射
        通过大典四级码 JOIN O*NET 参考表，附加国际职业描述
        输出: data/output/final/year={year}/level1_code={xx}/data.parquet
```

### 主要特性

| 特性 | 说明 |
|------|------|
| **四步完整流水线** | Step1 清洗 → Step2 LLM → Step3 向量匹配 → Step4 O\*NET，pipeline.py 统一调度 |
| **断点续跑** | 每个步骤独立 checkpoint，崩溃后自动从上次写盘位置恢复 |
| **自动重试** | CUDA OOM 等已知 vLLM 崩溃自动重试，默认最多 10 次 |
| **多 GPU 并行** | 设置 `NUM_GPUS=N` 即可按年份并行分配到多张卡 |
| **可选精排** | 启用 `use_rerank: true` 后，用 BGE-Reranker-v2-M3 对 top-k 候选重排序 |
| **匹配阈值** | `min_embed_score: 0.55`，低于此分数直接标记为未匹配，避免低质量结果 |
| **步骤间行数校验** | 自动对比前后步骤输出行数，检测静默丢数据 |
| **环境变量路径** | 通过 `PIPELINE_BASE_DIR` 环境变量切换数据根目录，无需改 config |
| **日志持久化** | 所有步骤的日志同时写入 `data/output/logs/`，便于事后排查 |
| **Hive 分区输出** | 按 `year` + `level1_code` 双层分区存储，便于下游 DuckDB/Spark 查询 |

---

## 目录结构

```
job2occ/
├── pipeline.py                  # 主调度器（多年/多卡并行）
├── requirements_cloud.txt       # Python 依赖清单
├── .gitignore
├── DEPLOYMENT_GUIDE.md          # 云端完整部署手册
│
├── step1_clean/
│   ├── step1_clean_content.py   # 数据清洗（去重 + 过滤）
│   └── config.yaml              # Step1 路径与清洗参数
│
├── step2_vllm/
│   ├── standardize_vllm.py      # LLM 岗位职责标准化
│   └── config.yaml              # Step2 路径与模型参数
│
├── step3_embedding/
│   ├── embed_and_match_gpu.py   # BGE-M3 向量匹配 + 可选 Reranker 精排
│   └── config.yaml              # Step3 路径与匹配参数
│
├── step4_onet/
│   ├── add_onet_mapping.py      # DuckDB SQL JOIN O*NET
│   └── config.yaml              # Step4 路径与字段配置
│
└── output/
    └── merge.py                 # Parquet 分区合并工具
```

---

## 技术栈

### 运行环境

| 项目 | 要求 |
|------|------|
| 操作系统 | Linux（云端 AutoDL 等） |
| Python | **3.10**（必须，3.12 与 vLLM 0.6.6 不兼容） |
| GPU | NVIDIA RTX 4090（24GB VRAM）或同等级 |
| 显存（Step2） | ~5GB（AWQ 量化后） |
| 显存（Step3） | ~10GB（float16 + 向量缓存） |
| 内存 | >= 64GB RAM |
| 磁盘 | >= 400GB（数据盘） |

### 核心依赖

| 库 | 版本 | 用途 |
|----|------|------|
| **vLLM** | 0.6.6 | LLM offline batch 推理（Step2）+ 向量 embedding（Step3） |
| **PyTorch** | 随 vLLM | GPU 计算基础 |
| **Transformers** | >=4.45.0 | 模型加载 |
| **FlagEmbedding** | >=1.2.0 | Step3 可选精排（BGE-Reranker-v2-M3） |
| **DuckDB** | >=0.10.0 | Step4 内存 SQL JOIN，以及下游数据查询 |
| **PyArrow** | >=14.0.0 | Parquet 读写 |
| **Pandas** | >=2.1.0 | 数据处理 |
| **NumPy** | >=1.26.0 | 向量运算 |
| **openpyxl** | >=3.1.0 | 读取职业分类大典 `.xlsx` |
| **PyYAML** | >=6.0 | 解析各步骤 `config.yaml` |
| **tqdm** | >=4.66.0 | 进度条 |

### 模型

| 模型 | 大小 | 用途 |
|------|------|------|
| `Qwen2.5-7B-Instruct-AWQ` | ~5GB | Step2：岗位职责文本抽取与标准化（AWQ 4-bit 量化） |
| `BAAI/bge-m3` | ~4.3GB | Step3：多语言稠密向量 embedding，余弦相似度职业匹配 |
| `BAAI/bge-reranker-v2-m3` | ~2.2GB | Step3（可选）：交叉编码器精排，提升匹配准确率 |

### 参考数据

| 文件 | 用途 |
|------|------|
| `职业分类大典2022_完整数据.xlsx` | Step3 匹配目标：8 大类 -> 四级职业名称与描述 |
| `dadian_to_onet_mapping.csv` | Step4 JOIN 键：大典四级码 -> O\*NET SOC 编码 |
| `onet_for_matching.csv` | Step4 附加信息：O\*NET 职业描述与任务文本 |

---

## 快速开始

### 1. 环境安装

```bash
# 创建 conda 环境（数据盘，避免占满系统盘）
conda create -p /root/autodl-tmp/envs/recruit python=3.10 -y
conda activate /root/autodl-tmp/envs/recruit

# 安装 vLLM（含 AWQ + embedding 支持）
pip install vllm==0.6.6

# 安装其他依赖
pip install -r requirements_cloud.txt
```

### 2. 运行 Pipeline

```bash
# 单年测试（推荐先跑）
python job2occ/pipeline.py --year 2021

# 全年顺序执行（单 GPU）
python job2occ/pipeline.py --all-years

# 多 GPU 并行（4 张卡，每卡处理不同年份）
NUM_GPUS=4 python job2occ/pipeline.py --all-years

# 断点续跑（step1/step2 已完成，从 step3 开始）
python job2occ/pipeline.py --year 2021 --start-from step3

# 只运行某一步
python job2occ/pipeline.py --year 2021 --only step4
```

### 4. 环境变量

```bash
# 切换数据根目录（默认 /root/autodl-tmp）
PIPELINE_BASE_DIR=/data/myproject python job2occ/pipeline.py --year 2021
```

---

## 数据规模

| 年份范围 | 输入大小 | 预计输出 |
|---------|---------|---------|
| 2015-2024（10 年） | ~173GB CSV | ~80-120GB Parquet |

### 参考吞吐量（RTX 4090，AWQ + prefix caching）

| 步骤 | 速度 |
|------|------|
| Step1：数据清洗 | ~50,000+ 条/秒（CPU） |
| Step2：vLLM AWQ 标准化 | ~30-50 条/秒 |
| Step3：BGE-M3 向量匹配 | ~500+ 条/秒 |
| Step4：DuckDB JOIN | 秒级（按年份） |

---

## 输出格式

输出以 Hive 分区 Parquet 存储，可直接用 DuckDB 或 Spark 查询：

```
data/output/final/
├── year=2015/
│   ├── level1_code=01/data.parquet
│   ├── level1_code=02/data.parquet
│   ├── level1_code=00/data.parquet    (未匹配)
│   └── ...（共 8 个大类 + 未匹配）
├── year=2016/
│   └── ...
└── year=2024/
    └── ...
```

### 合并分区

```bash
# 合并单年
python job2occ/output/merge.py --year 2021

# 合并全部年份
python job2occ/output/merge.py --all-years
```

### 查询示例

```python
import duckdb

con = duckdb.connect()

# 按年份统计各职业大类数量
con.execute("""
    SELECT year, level1, COUNT(*) as cnt
    FROM read_parquet('data/output/final/**/*.parquet', hive_partitioning=true)
    GROUP BY year, level1
    ORDER BY year, cnt DESC
""").df()

# 跨年份追踪计算机职业趋势（O*NET 15-xxxx）
con.execute("""
    SELECT year, onet_title, COUNT(*) as cnt
    FROM read_parquet('data/output/final/**/*.parquet', hive_partitioning=true)
    WHERE onet_code LIKE '15-%'
    GROUP BY year, onet_title
    ORDER BY year, cnt DESC
""").df()
```

---

## 日志与排查

所有步骤的日志自动写入 `$PIPELINE_BASE_DIR/data/output/logs/`：

```
logs/
├── pipeline_20250303_140000.log   # 总调度日志
├── step1_2021.log
├── step2_2021.log
├── step3_2021.log
└── step4_2021.log
```

---

## 部署说明

完整的云端部署步骤（文件上传、conda 环境配置、模型下载、验证与全量运行）见 [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)。
