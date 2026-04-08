# 云端部署操作手册

## 概览

| 项目 | 说明 |
|------|------|
| 目标环境 | 云端 Linux 服务器，RTX 4090（24GB VRAM） |
| 总上传数据量 | 约 **179GB**（原始数据 173GB + 模型 ~6GB + 参考文件 <1GB） |
| 总输出预估 | 约 80~120GB（Parquet 压缩后） |
| 建议磁盘空间 | **≥ 400GB**（数据磁盘：autodl-tmp） |
| 建议内存 | ≥ 64GB RAM |
| Python 版本 | **3.10**（必须，3.12 不兼容 vLLM wheel） |

### 架构说明

- **Step 1**：数据清洗（去重 + 过滤过短描述），纯 CPU，已集成到 pipeline 统一调度
- **Step 2**：vLLM offline batch 模式（`LLM.generate()`），AWQ 量化模型，prefix caching 加速
- **Step 3**：vLLM embedding 模式（`LLM(task='embedding')`）+ 可选 BGE-Reranker-v2-M3 精排，GPU 矩阵计算
- **Step 4**：DuckDB 内存 SQL JOIN，写出 Hive 分区 Parquet
- **单一 conda 环境**：所有步骤共用 `recruit` 环境，无环境切换
- **环境变量**：通过 `PIPELINE_BASE_DIR` 切换数据根目录（默认 `/root/autodl-tmp`）
- **日志持久化**：所有步骤日志写入 `data/output/logs/`

---

## 第一步：上传文件清单

### 1.1 需要上传的文件

#### A. 脚本（必须上传）

| 本地路径 | 云端路径 |
|---------|---------|
| `job2occ/` （整个文件夹）| `/root/autodl-tmp/job2occ/` |

共约 10 个文件，体积 < 100KB，直接 rsync 即可。

#### B. 模型（必须上传）

| 本地路径 | 云端路径 | 大小 |
|---------|---------|------|
| `models/qwen2.5-7b-instruct-awq/` | `/root/autodl-tmp/models/qwen2.5-7b-instruct-awq/` | **~5GB** |
| `models/bge-m3/` | `/root/autodl-tmp/models/bge-m3/` | **~4.3GB** |

> **说明**：Step 2 使用 AWQ 量化版 Qwen2.5-7B，VRAM 占用从 14GB 降至 ~5GB，推理更快。
> bge-reranker-v2-m3 为可选精排模型（`use_rerank: false` 时不需要上传，启用时需额外上传 ~2.2GB）。
> 如果模型未从本地上传，也可直接在云端从 ModelScope 下载（见 §2.2）。

#### C. 原始数据（必须上传，体积最大）

> **说明**：Step1（数据清洗）已集成到 pipeline 统一调度。如果上传的是**未清洗**的原始 CSV，放到 `data/raw/`，pipeline 会自动跑 Step1 清洗后输出到 `data/input/`。如果上传的是**已清洗**的 CSV（文件名含 `_step1_clean`），直接放到 `data/input/`，运行时用 `--start-from step2` 跳过 Step1。

| 本地路径 | 云端路径 | 大小 |
|---------|---------|------|
| `recruitment_infos_content_2015.csv`（或 `*_step1_clean.csv`） | `/root/autodl-tmp/data/raw/`（或 `data/input/`） | 3.0GB |
| `recruitment_infos_content_2016.csv` | 同上 | 13GB |
| `recruitment_infos_content_2017.csv` | 同上 | 24GB |
| `recruitment_infos_content_2018.csv` | 同上 | 39GB |
| `recruitment_infos_content_2019.csv` | 同上 | 17GB |
| `recruitment_infos_content_2020.csv` | 同上 | 15GB |
| `recruitment_infos_content_2021.csv` | 同上 | 17GB |
| `recruitment_infos_content_2022.csv` | 同上 | 18GB |
| `recruitment_infos_content_2023.csv` | 同上 | 14GB |
| `recruitment_infos_content_2024.csv` | 同上 | 13GB |
| **合计** | | **~173GB** |

#### D. 参考数据（必须上传，体积小）

| 本地路径 | 云端路径 | 大小 |
|---------|---------|------|
| `step3_bge_m3_embedding/职业分类大典2022_完整数据.xlsx` | `/root/autodl-tmp/data/reference/` | 472KB |
| `step4_O*NET/mapping_results/dadian_to_onet_mapping.csv` | `/root/autodl-tmp/data/reference/` | 308KB |
| `step4_O*NET/vectorized/onet_for_matching.csv` | `/root/autodl-tmp/data/reference/` | 3.0MB |

#### E. 可选上传：Reranker 模型

如需启用 Step3 精排功能（`use_rerank: true`），需额外上传 Reranker 模型：

| 本地路径 | 云端路径 | 大小 |
|---------|---------|------|
| `models/bge-reranker-v2-m3/` | `/root/autodl-tmp/models/bge-reranker-v2-m3/` | **~2.2GB** |

> 全量 1.6 亿条数据开启精排会显著增加耗时，建议仅在小规模数据或对准确率要求极高时启用。

#### F. 不需要上传

- `venv/` — 云端重新建 conda 环境
- `test_output/` — 本地测试结果
- `step3_bge_m3_embedding/cache/` — 云端会自动重建（精度不同）
- 各种本地 `.md`、`.sh`、本地版 `pipeline.py` 等

---

### 1.2 上传命令

在**本地 Mac** 终端执行（替换 `<SERVER_IP>` 和 `<SSH_USER>`）：

```bash
# 设置服务器信息
SERVER="<SSH_USER>@<SERVER_IP>"
REMOTE="/root/autodl-tmp"

# 在服务器上创建目录结构
ssh $SERVER "mkdir -p $REMOTE/{job2occ,models,data/{raw,input,reference,output/{standardized,matched,final,checkpoints/{step2,step3,step4},cache,logs}}}"

# 上传脚本（秒级）
rsync -avh job2occ/ $SERVER:$REMOTE/job2occ/

# 上传参考数据（秒级）
scp "step3_bge_m3_embedding/职业分类大典2022_完整数据.xlsx" $SERVER:$REMOTE/data/reference/
scp "step4_O*NET/mapping_results/dadian_to_onet_mapping.csv" $SERVER:$REMOTE/data/reference/
scp "step4_O*NET/vectorized/onet_for_matching.csv" $SERVER:$REMOTE/data/reference/

# 上传模型（约 9GB，视带宽需要数十分钟~数小时）
rsync -avh --progress models/qwen2.5-7b-instruct-awq/ $SERVER:$REMOTE/models/qwen2.5-7b-instruct-awq/
rsync -avh --progress models/bge-m3/ $SERVER:$REMOTE/models/bge-m3/

# 上传原始数据（约 173GB，后台运行防止 SSH 超时）
nohup rsync -avh --progress \
    -e "ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=10" \
    step1_clean/recruitment_infos_content_*.csv \
    $SERVER:$REMOTE/data/input/ \
    > ~/rsync_upload.log 2>&1 &

# 查看上传进度
tail -f ~/rsync_upload.log
```

> **大文件上传建议**：173GB 通过家庭宽带上传可能需要数天。优先考虑：
> - 云服务商的对象存储中转（上传到 OSS/S3 → 云服务器内网下载，速度 100MB/s+）
> - 询问云服务商是否支持本地硬盘寄送

---

## 第二步：服务器环境配置

登录服务器后，按以下顺序执行。

### 2.1 确认 GPU 和 CUDA

```bash
# 确认 GPU 型号和 VRAM
nvidia-smi

# 确认 CUDA 版本
nvidia-smi | grep "CUDA Version"
```

预期输出：
```
NVIDIA RTX 4090, 24564 MiB
CUDA Version: 12.x
```

### 2.2 创建 conda 环境（Python 3.10，装在数据盘）

> **重要**：conda 环境必须装在数据盘（`/root/autodl-tmp`），避免 30GB 系统盘被 torch/CUDA 库塞满。

```bash
# 初始化 conda（如果新终端找不到 conda activate 命令）
source /root/miniconda3/etc/profile.d/conda.sh

# 创建 Python 3.10 环境到数据盘
conda create -p /root/autodl-tmp/envs/recruit python=3.10 -y

# 激活（后续所有命令在此环境下执行）
conda activate /root/autodl-tmp/envs/recruit

# 将激活命令加入 .bashrc（新终端自动激活）
echo 'source /root/miniconda3/etc/profile.d/conda.sh' >> ~/.bashrc
echo 'conda activate /root/autodl-tmp/envs/recruit' >> ~/.bashrc
```

### 2.3 下载 AWQ 量化模型（如未从本地上传）

如果模型已从本地上传，跳过此步骤。

```bash
# 从 ModelScope 下载 AWQ 模型（约 5GB，速度快）
pip install modelscope

python -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download(
    'Qwen/Qwen2.5-7B-Instruct-AWQ',
    cache_dir='/root/autodl-tmp/models',
    local_dir='/root/autodl-tmp/models/qwen2.5-7b-instruct-awq'
)
"

# BGE-M3 同样可从 ModelScope 下载（约 4.3GB）
python -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download(
    'BAAI/bge-m3',
    cache_dir='/root/autodl-tmp/models',
    local_dir='/root/autodl-tmp/models/bge-m3'
)
"

# 可选：BGE-Reranker-v2-M3（约 2.2GB，仅启用精排时需要）
python -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download(
    'BAAI/bge-reranker-v2-m3',
    cache_dir='/root/autodl-tmp/models',
    local_dir='/root/autodl-tmp/models/bge-reranker-v2-m3'
)
"
```

### 2.4 安装依赖

```bash
# 确保在 recruit 环境下
conda activate /root/autodl-tmp/envs/recruit

# 安装 vLLM 0.6.6（含 AWQ + embedding 支持）
pip install vllm==0.6.6

# 安装其他依赖
pip install -r /root/autodl-tmp/job2occ/requirements_cloud.txt
```

### 2.5 验证安装

```bash
# 验证 vLLM 版本
python -c "import vllm; print('vLLM:', vllm.__version__)"
# 预期: vLLM: 0.6.6

# 验证 CUDA 可用
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
# 预期: CUDA: True
#       NVIDIA GeForce RTX 4090

# 验证其他依赖
python -c "
import transformers, pandas, pyarrow, duckdb, yaml, tqdm, openpyxl
print('transformers:', transformers.__version__)
print('duckdb:', duckdb.__version__)
print('pyarrow:', pyarrow.__version__)
print('所有依赖安装成功')
"

# 快速验证 Step2 模型加载（约 1 分钟）
python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='/root/autodl-tmp/models/qwen2.5-7b-instruct-awq', dtype='auto', max_model_len=512)
out = llm.generate(['你好'], SamplingParams(max_tokens=10))
print('Step2 模型 OK:', out[0].outputs[0].text[:30])
del llm
"

# 快速验证 Step3 模型加载（约 30 秒）
python -c "
from vllm import LLM
llm = LLM(model='/root/autodl-tmp/models/bge-m3', task='embedding', dtype='float16', gpu_memory_utilization=0.45)
out = llm.embed(['测试文本'])
print('Step3 模型 OK, embedding dim:', len(out[0].outputs.embedding))
del llm
"
# 预期: Step3 模型 OK, embedding dim: 1024
```

---

## 第三步：配置验证

### 3.1 确认目录结构

```bash
tree /root/autodl-tmp/ -L 3 --dirsfirst
```

预期结构：
```
/root/autodl-tmp/
├── job2occ/
│   ├── pipeline.py
│   ├── test_run.py
│   ├── requirements_cloud.txt
│   ├── step1_clean/
│   │   ├── step1_clean_content.py
│   │   └── config.yaml
│   ├── step2_vllm/
│   │   ├── standardize_vllm.py
│   │   └── config.yaml
│   ├── step3_embedding/
│   │   ├── embed_and_match_gpu.py
│   │   └── config.yaml
│   ├── step4_onet/
│   │   ├── add_onet_mapping.py
│   │   └── config.yaml
│   └── output/
│       └── merge.py
├── envs/
│   └── recruit/              <- conda 环境
├── models/
│   ├── qwen2.5-7b-instruct-awq/
│   ├── bge-m3/
│   └── bge-reranker-v2-m3/   <- 可选（启用精排时）
└── data/
    ├── raw/                   <- 原始 CSV（Step1 输入）
    ├── input/                 <- 清洗后 CSV（Step2 输入）
    │   ├── recruitment_infos_content_2015_step1_clean.csv
    │   └── ... (共 10 个年份)
    ├── reference/
    │   ├── 职业分类大典2022_完整数据.xlsx
    │   ├── dadian_to_onet_mapping.csv
    │   └── onet_for_matching.csv
    └── output/                <- 运行后自动填充
        ├── logs/              <- 日志文件
        ├── checkpoints/       <- 断点文件（step2/step3/step4）
        ├── standardized/      <- Step2 输出
        ├── matched/           <- Step3 输出
        └── final/             <- Step4 输出（Hive 分区）
```

### 3.2 检查 config 路径是否正确

```bash
python3 -c "
import yaml, os
configs = [
    '/root/autodl-tmp/job2occ/step1_clean/config.yaml',
    '/root/autodl-tmp/job2occ/step2_vllm/config.yaml',
    '/root/autodl-tmp/job2occ/step3_embedding/config.yaml',
    '/root/autodl-tmp/job2occ/step4_onet/config.yaml',
]
path_keys = [
    'input_dir', 'output_dir', 'llm_model', 'embed_model', 'rerank_model',
    'occupation_path', 'mapping_table_path', 'onet_detail_path',
]
for cfg_path in configs:
    print(f'\n--- {cfg_path.split(\"/\")[-2]} ---')
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    for key in path_keys:
        if key in cfg:
            exists = os.path.exists(cfg[key])
            status = '✓' if exists else '✗ 不存在!'
            print(f'{status} [{key}] {cfg[key]}')
"
```

> **提示**：如果使用 `PIPELINE_BASE_DIR` 环境变量，路径会在运行时自动替换，config 中的硬编码路径无需修改。

如果有 `✗` 路径，直接编辑对应 config.yaml 修正。

---

## 第四步：小样本测试（强烈建议先跑）

在跑全量 1.6 亿条之前，先用 1000 条样本验证速度和准确度。

```bash
cd /root/autodl-tmp

# 小样本速度与准确度测试（约 5~10 分钟）
python job2occ/test_run.py --limit 1000
```

测试脚本会：
1. 从 2015 年数据抽取前 1000 条
2. 顺序运行 Step2 → Step3 → Step4
3. 输出速度报告（条/秒、全量时间推算）
4. 随机抽 20 条供人工核查匹配质量

预期速度参考（RTX 4090，AWQ 量化 + prefix caching）：
```
Step2 (vLLM AWQ 标准化):   ~30-50 条/秒
Step3 (vLLM BGE-M3 匹配):  ~500+ 条/秒
Step4 (DuckDB JOIN):        <1 秒（小样本）
```

---

## 第五步：单年测试（端到端验证）

小样本测试通过后，用 2021 年单年验证完整流程。

```bash
cd /root/autodl-tmp

# 单年端到端测试
python job2occ/pipeline.py --year 2021
```

预期日志（offline batch 模式，无 HTTP 服务器启动）：
```
云端 Pipeline 启动
执行步骤: ['step1', 'step2', 'step3', 'step4']
可用 GPU 数: 1
PIPELINE_BASE_DIR: /root/autodl-tmp
==================================================
开始处理年份: 2021（GPU 0）
--- STEP1 | 年份 2021 ---
  清洗完成: 保留 17,000,000 | 去重 500,000 | 过短 300,000
--- STEP2 | 年份 2021 ---
  Loading model: qwen2.5-7b-instruct-awq ...
  Step2 2021: 100%|████████| 1700万/1700万 [05:00<00:00, 50条/s]
行数校验: step1→step2 year=2021 | 一致 (17,000,000 行)
--- STEP3 | 年份 2021 ---
  Loading model: bge-m3 ...
  Step3 2021: 100%|████████| 1700万/1700万 [01:30<00:00, 500条/s]
行数校验: step2→step3 year=2021 | 一致 (17,000,000 行)
--- STEP4 | 年份 2021 ---
  DuckDB JOIN 完成，写出 Hive 分区...
✓ 年份 2021 全部完成，耗时 X.X 小时
```

验证输出：
```bash
# 检查输出文件是否存在
ls -lh data/output/standardized/year=2021/
ls -lh data/output/matched/year=2021/
ls -lh data/output/final/year=2021/

# 用 DuckDB 抽查数据质量
python3 -c "
import duckdb
# 抽查 Step2 输出
df = duckdb.query(\"SELECT standardized_job_description FROM 'data/output/standardized/year=2021/data.parquet' LIMIT 5\").df()
print('=== Step2 输出示例 ===')
for v in df['standardized_job_description']:
    print(v[:100])

# 抽查 Step3 匹配
df2 = duckdb.query(\"SELECT match_occupation, embed_score, is_match FROM 'data/output/matched/year=2021/data.parquet' LIMIT 10\").df()
print('\n=== Step3 匹配示例 ===')
print(df2.to_string())

# 抽查 Step4 最终输出
df3 = duckdb.query(\"SELECT year, level1, onet_code, onet_title FROM 'data/output/final/**/*.parquet' WHERE year='2021' LIMIT 5\").df()
print('\n=== Step4 最终输出示例 ===')
print(df3.to_string())
"
```

---

## 第六步：全量运行

单年测试通过后，开始跑全部年份。

### 方案 A：单卡顺序（1 张 4090）

```bash
# 在 tmux 中运行，防止 SSH 断线中断
tmux new-session -s pipeline

conda activate /root/autodl-tmp/envs/recruit
cd /root/autodl-tmp
python job2occ/pipeline.py --all-years

# 断开 SSH 后，随时用以下命令重新连接查看进度
tmux attach -t pipeline
```

预估时间：约 **4~6 天**（取决于实际测速结果）

### 方案 B：多卡并行（多张 4090）

```bash
tmux new-session -s pipeline

conda activate /root/autodl-tmp/envs/recruit
cd /root/autodl-tmp

# 4 卡并行，每卡处理不同年份
NUM_GPUS=4 python job2occ/pipeline.py --all-years
```

预估时间：约 **1~2 天**（4 卡并行）

> **说明**：多卡并行时，每张卡的子进程独立加载 vLLM 模型（offline batch 模式），通过 `CUDA_VISIBLE_DEVICES` 隔离，无需分配不同端口。

### 断点续跑

如果某年份中途失败，只需重新运行，Step2 脚本会自动从断点继续：

```bash
# 只重跑某一年
python job2occ/pipeline.py --year 2018

# 如果 step2 已完成，从 step3 开始
python job2occ/pipeline.py --year 2018 --start-from step3

# 只跑某个步骤
python job2occ/pipeline.py --year 2018 --only step4
```

---

## 第七步：结果查询

Pipeline 完成后，输出结构如下：

```
data/output/final/
├── year=2015/
│   ├── level1=专业技术人员/data.parquet
│   ├── level1=办事人员和有关人员/data.parquet
│   ├── level1=生产制造及有关人员/data.parquet
│   ├── level1=未匹配/data.parquet
│   └── ... (8 个大类)
├── year=2016/
│   └── ...
└── year=2024/
    └── ...
```

用 DuckDB 查询示例：

```python
import duckdb

con = duckdb.connect()

# 查某年某职业类别的数量分布
con.execute("""
    SELECT year, level1, COUNT(*) as cnt
    FROM read_parquet('data/output/final/**/*.parquet', hive_partitioning=true)
    WHERE year = '2021'
    GROUP BY year, level1
    ORDER BY cnt DESC
""").df()

# 跨年份查特定 O*NET 职业的趋势
con.execute("""
    SELECT year, onet_title, COUNT(*) as cnt
    FROM read_parquet('data/output/final/**/*.parquet', hive_partitioning=true)
    WHERE onet_code LIKE '15-%'   -- 计算机相关职业
    GROUP BY year, onet_title
    ORDER BY year, cnt DESC
""").df()

# 查匹配质量分布
con.execute("""
    SELECT
        year,
        is_match,
        COUNT(*) as cnt,
        AVG(embed_score) as avg_score
    FROM read_parquet('data/output/final/**/*.parquet', hive_partitioning=true)
    GROUP BY year, is_match
    ORDER BY year, is_match
""").df()
```

---

## 常见问题

### Q1: vLLM 模型加载失败

```bash
# 手动测试模型加载，查看完整报错
python -c "
from vllm import LLM
llm = LLM(model='/root/autodl-tmp/models/qwen2.5-7b-instruct-awq', dtype='auto', max_model_len=2048)
print('加载成功')
"
```

常见原因：
- VRAM 不足：`nvidia-smi` 确认空闲 VRAM ≥ 8GB（AWQ 约 5GB）
- 模型文件不完整：`ls -la /root/autodl-tmp/models/qwen2.5-7b-instruct-awq/`，确认有 `.safetensors` 文件
- conda 环境未激活：确认 `conda activate /root/autodl-tmp/envs/recruit`

### Q2: Step3 OOM（显存不足）

编辑 `step3_embedding/config.yaml`：
```yaml
query_chunk_size: 100000  # 从 250000 降至 100000
gpu_memory_utilization: 0.35  # 从 0.45 降低
```

### Q3: 上传中断，数据不完整

```bash
# rsync 天然支持断点续传，重新执行同一命令即可
rsync -avh --progress step1_clean/recruitment_infos_content_2021_step1_clean.csv \
    $SERVER:/root/autodl-tmp/data/input/
# rsync 会跳过已完整传输的文件，只补传差异部分
```

### Q4: 查看实时进度

```bash
# 查看所有步骤的断点状态
python3 -c "
import json, glob
for f in sorted(glob.glob('data/output/checkpoints/*/*.json')):
    with open(f) as fh:
        state = json.load(fh)
    done = state.get('rows_done', 0)
    complete = state.get('complete', False)
    status = '已完成' if complete else f'进行中 ({done:,} 行)'
    print(f'{f}: {status}')
"

# 查看日志（实时跟踪）
tail -f data/output/logs/step2_2021.log

# 查看 GPU 实时利用率
watch -n 2 nvidia-smi
```

### Q5: 磁盘空间不足

```bash
# 查看磁盘使用
df -h

# 查看各目录大小
du -sh data/output/*/
```

如果 `standardized/` 和 `matched/` 已完成且 `final/` 也已生成，可删除中间层：
```bash
# 确认 final 完整后再删（谨慎操作）
rm -rf data/output/standardized/year=2015/
rm -rf data/output/matched/year=2015/
```

### Q6: conda activate 在新终端失效

```bash
# 手动初始化 conda，然后激活
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/recruit

# 确认已写入 .bashrc（应该已经有这两行）
grep conda ~/.bashrc
```
