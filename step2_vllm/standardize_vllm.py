#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
职位描述标准化工具 - 云端 vLLM Offline Batch 版本

改动说明（相比 HTTP API 版本）：
- LLM 推理：httpx 异步 HTTP → vLLM LLM Python API（offline batch）
- 批处理：asyncio.Semaphore → vLLM 内部连续批处理（GPU 利用率 95%+）
- 不再需要启动 HTTP 服务，直接在进程内加载模型
- 断点续传：按年份独立的 checkpoint JSON（逻辑不变）
- 输出格式：Parquet（逻辑不变）

使用方式：
    python standardize_vllm.py --config config.yaml --year 2021
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# =============================================================================
# 提示词模板（与本地版本完全一致）
# =============================================================================

SYSTEM_PROMPT = (
    '你是"职业描述标准化"助手。请把输入的招聘职位描述改写成标准化职业描述，'
    '风格参考《职业分类大典2022》的 main_task：简洁、客观、任务导向。'
)

# 固定前缀：包含指令 + few-shot 示例，所有请求共享，用于 prefix caching
# {example_tasks} 在 load_model() 时一次性填充，之后不再变化
USER_PREFIX_TEMPLATE = """请把下面的职位描述标准化改写，要求：
1) 仅保留"岗位职责/工作内容"，删除任职要求、福利、公司介绍、口号等；
2) 不新增不存在的信息，不推断；保持原意与关键职责；
3) 用中文输出，使用"1. …；2. …；3. …"格式，写出2-5条任务；
4) 输出仅包含标准化职业描述文本，不要加解释、标题或其他内容；
5) 只要存在明确的任务或职责描述，就必须抽取并输出；
6) 如果输入为空或完全无法识别岗位职责，输出"无法识别"。

以下是《职业分类大典2022》main_task 的标准表达示例（仅用于风格参考）：
{example_tasks}

职位描述：
"""

# 变化后缀：每条请求不同，仅包含实际职位描述文本（由调用方拼接）


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class Config:
    """云端配置类"""

    # 输入输出
    input_dir: str
    input_filename_template: str
    output_dir: str
    resume_state_dir: str

    # 字段配置
    text_column: str
    output_column: str

    # vLLM offline batch 配置
    llm_model: str
    tensor_parallel_size: int
    max_model_len: int
    max_num_seqs: int              # 最大并发序列数（直接决定 GPU batch 大小，建议 256-512）

    # 批处理与写盘
    llm_batch_size: int    # 每次调用 llm.generate() 的条数（影响 GPU batch 大小，建议 50000+）
    write_chunk_size: int  # 每次写盘的条数（影响断点粒度，建议 10000）

    # LLM 生成参数
    llm_temperature: float
    llm_top_p: float
    llm_top_k: int
    llm_num_predict: int
    max_input_chars: int

    # Few-shot 示例
    example_tasks: List[str]

    # 职责提取规则
    duty_headers: List[str]
    negative_headers: List[str]
    negative_phrases: List[str] = field(default_factory=list)   # 云端轻量版不再使用
    positive_verbs: List[str] = field(default_factory=list)     # 云端轻量版不再使用

    # 额外字段
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # 环境变量替换路径前缀
        base_dir = os.environ.get("PIPELINE_BASE_DIR", "")
        if base_dir:
            for key in ("input_dir", "output_dir", "resume_state_dir", "llm_model"):
                if key in data and isinstance(data[key], str):
                    data[key] = data[key].replace("/root/autodl-tmp", base_dir)
        field_names = {f.name for f in fields(cls)}
        core = {k: v for k, v in data.items() if k in field_names}
        extra = {k: v for k, v in data.items() if k not in field_names}
        config = cls(**core)
        config.extra = extra
        return config

    def input_path(self, year: str) -> str:
        filename = self.input_filename_template.format(year=year)
        return os.path.join(self.input_dir, filename)

    def output_parquet_dir(self, year: str) -> str:
        return os.path.join(self.output_dir, f"year={year}")

    def checkpoint_path(self, year: str) -> str:
        return os.path.join(self.resume_state_dir, f"step2_{year}.json")


# =============================================================================
# 统计类
# =============================================================================

@dataclass
class ProcessingStats:
    total: int = 0
    processed: int = 0
    success: int = 0
    failed: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def speed(self) -> float:
        return self.processed / self.elapsed if self.elapsed > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"已处理: {self.processed}/{self.total} | "
            f"成功: {self.success} | 失败: {self.failed} | "
            f"速度: {self.speed:.1f} 条/秒"
        )


# =============================================================================
# 日志配置
# =============================================================================

def setup_logging(year: str = "") -> logging.Logger:
    logger = logging.getLogger(f"job2occ.step2_{year}")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)

    # 日志文件持久化
    base_dir = os.environ.get("PIPELINE_BASE_DIR", "/root/autodl-tmp")
    log_dir = Path(base_dir) / "data" / "output" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / f"step2_{year}.log", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


# =============================================================================
# 职责提取与格式化（与本地版本完全一致）
# =============================================================================

class DutyExtractor:
    """
    职责文本提取器 — 轻量版（云端 vLLM 专用）

    设计思路：
    本地 transformers 版需要在 CPU 上逐条推理（~0.5条/s），因此依赖重度 regex
    过滤来缩短输入 token 数。但 vLLM batch 推理的瓶颈在 decode 阶段（自回归生成），
    而非 prefill，且 prefix caching 已消除共享前缀的开销。因此：

    1. 仅做"段落级"切割：找到 duty_headers → negative_headers 之间的段落
    2. 不做逐行 negative_phrases 过滤（这些词如 "熟悉""培训""能力" 大量出现在
       合法职责行中，过滤会导致 ~6% 准确率损失）
    3. 如果没有找到 duty_headers，直接返回原始文本 — 让 7B LLM 自行区分
    """

    def __init__(self, config: Config):
        self._duty_headers = config.duty_headers
        self._negative_headers = config.negative_headers

    def _split_lines(self, text: str) -> List[str]:
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
        for header in self._duty_headers + self._negative_headers:
            text = re.sub(
                rf"(?<!\n){re.escape(header)}\s*[:：]?",
                f"\n{header}：",
                text,
            )
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return lines

    def _is_header(self, line: str, headers: List[str]) -> bool:
        for header in headers:
            if re.search(rf"^\s*{re.escape(header)}\s*[:：]?", line):
                return True
        return False

    def _strip_header(self, line: str, headers: List[str]) -> str:
        for header in headers:
            line = re.sub(rf"^\s*{re.escape(header)}\s*[:：]?\s*", "", line)
        return line.strip()

    def extract(self, text: str) -> str:
        """
        轻量提取：仅按 header 做段落级切割，不做逐行过滤。
        - 找到 duty_header → 截至 negative_header 之间的内容
        - 找不到 duty_header → 返回空字符串，由调用方回退到原文
        """
        lines = self._split_lines(text)
        if not lines:
            return ""

        # 寻找职责段落起始
        header_index = None
        for idx, line in enumerate(lines):
            if self._is_header(line, self._duty_headers):
                header_index = idx
                break

        if header_index is not None:
            collected: List[str] = []
            for line in lines[header_index:]:
                if self._is_header(line, self._negative_headers):
                    break
                line = self._strip_header(line, self._duty_headers)
                if line:
                    collected.append(line)
            if collected:
                return "\n".join(collected)

        # 无 duty_header：仅截断到 negative_header（如有）
        # 不做 negative_phrases 逐行过滤，让 LLM 自行判断
        collected = []
        for line in lines:
            if self._is_header(line, self._negative_headers):
                break
            collected.append(line)
        return "\n".join(collected)


def format_numbered_output(text: str) -> str:
    """格式化编号输出（与本地版本完全一致）"""
    text = text.strip()
    if not text or text == "无法识别":
        return text
    text = re.sub(r"(?<!^)(?<!\n)\s*(\d+)[\.\、\)）]\s*", r"\n\1. ", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    formatted = []
    for line in lines:
        line = re.sub(r"^(\d+)[\.\、\)）]\s*", r"\1. ", line)
        if not line.endswith(("；", ";", "。", ".", "！", "!", "？", "?")):
            line += "；"
        formatted.append(line)
    return "\n".join(formatted)


# =============================================================================
# 断点管理
# =============================================================================

class CheckpointManager:
    """按年份独立的断点管理"""

    def __init__(self, path: str, logger: logging.Logger):
        self.path = Path(path)
        self.logger = logger
        self._state: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.path.exists():
            try:
                state = json.loads(self.path.read_text(encoding="utf-8"))
                self.logger.info(f"已加载断点: {self.path} (已完成 {state.get('rows_done', 0):,} 行)")
                return state
            except Exception as e:
                self.logger.warning(f"断点文件损坏，从头开始: {e}")
        return {"rows_done": 0, "chunks_written": 0}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    @property
    def rows_done(self) -> int:
        return self._state.get("rows_done", 0)

    @property
    def chunks_written(self) -> int:
        return self._state.get("chunks_written", 0)

    def update(self, rows_done: int, chunks_written: int):
        self._state["rows_done"] = rows_done
        self._state["chunks_written"] = chunks_written
        self.save()

    def mark_complete(self):
        self._state["complete"] = True
        self.save()

    @property
    def is_complete(self) -> bool:
        return self._state.get("complete", False)


# =============================================================================
# vLLM Offline Batch 处理器
# =============================================================================

class OfflineBatchProcessor:
    """
    使用 vLLM LLM Python API 进行 offline batch 推理。
    GPU 利用率可达 95%+，无 HTTP 开销。
    """

    def __init__(self, config: Config, extractor: DutyExtractor, logger: logging.Logger):
        self.config = config
        self.extractor = extractor
        self.logger = logger
        self._example_tasks_str = self._format_examples()
        self._llm = None

    def _format_examples(self) -> str:
        if not self.config.example_tasks:
            return "（无）"
        return "\n".join(
            f"示例{i+1}：{task}"
            for i, task in enumerate(self.config.example_tasks)
        )

    def load_model(self):
        """
        加载 vLLM 模型，预先构建固定前缀字符串。

        vLLM prefix caching 对字符串输入同样有效：
        它在内部 tokenize 后对 block hash 做匹配，只要所有请求的字符串前缀相同，
        KV cache 即可复用。传字符串比在 Python 侧串行 tokenize 效率更高
        （vLLM 内部用 C++ 批量 tokenize，比 Python 循环快 10x+）。
        """
        from vllm import LLM, SamplingParams
        self.logger.info(f"正在加载模型: {self.config.llm_model}")
        self._llm = LLM(
            model=self.config.llm_model,
            dtype="auto",                  # AWQ 模型自动选择量化精度；bfloat16 模型也兼容
            max_model_len=self.config.max_model_len,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=0.90,
            max_num_seqs=self.config.max_num_seqs,   # 最大并发序列数，直接决定 GPU batch 大小
            trust_remote_code=False,
            enable_prefix_caching=True,
        )
        self._sampling_params = SamplingParams(
            temperature=self.config.llm_temperature,
            top_p=self.config.llm_top_p,
            top_k=self.config.llm_top_k,
            max_tokens=self.config.llm_num_predict,
        )

        # 预先构建固定前缀字符串（只做一次）
        # 策略：用一个占位符 job_desc 生成完整 prompt，找到占位符位置，
        # 拆出 prefix（占位符之前）和 suffix（占位符之后）。
        # 每条请求的 prompt = _prompt_prefix + job_desc_text + _prompt_suffix
        # prefix 完全相同 → vLLM prefix cache 自动命中
        tokenizer = self._llm.get_tokenizer()
        user_prefix_text = USER_PREFIX_TEMPLATE.format(
            example_tasks=self._example_tasks_str,
        )
        _PLACEHOLDER = "\x00JOBDESC\x00"   # 不可能出现在真实数据中的占位符
        messages_full = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prefix_text + _PLACEHOLDER},
        ]
        full_prompt: str = tokenizer.apply_chat_template(
            messages_full,
            tokenize=False,
            add_generation_prompt=True,
        )
        assert _PLACEHOLDER in full_prompt, \
            f"占位符未出现在 prompt 中，chat template 可能对内容做了转义：{full_prompt[:200]}"

        split_pos = full_prompt.index(_PLACEHOLDER)
        self._prompt_prefix: str = full_prompt[:split_pos]
        self._prompt_suffix: str = full_prompt[split_pos + len(_PLACEHOLDER):]

        prefix_token_count = len(tokenizer.encode(self._prompt_prefix))
        self.logger.info(
            f"✓ 模型加载完成 | 固定前缀 {prefix_token_count} tokens（将被 prefix cache 复用）"
        )

    def _build_prompt(self, job_desc: str) -> str:
        """
        构建单条请求的完整 prompt 字符串。
        = _prompt_prefix（所有请求相同）+ job_desc 文本 + _prompt_suffix（generation prompt）
        prefix 完全相同 → vLLM prefix cache 自动命中。
        """
        raw = job_desc.strip()
        extracted = self.extractor.extract(raw) or raw
        truncated = extracted[: self.config.max_input_chars]
        return self._prompt_prefix + truncated + self._prompt_suffix

    def process_batch(self, texts: List[str]) -> List[Tuple[str, bool]]:
        """
        批量推理。传入字符串列表，vLLM 内部批量 tokenize（C++ 实现，远快于 Python 循环）。
        所有请求共享相同的固定前缀字符串，prefix cache 自动命中。

        返回: List of (output_text, success)
            success=True  — LLM 正常输出了标准化职责
            success=False — 空输入 / LLM 输出"无法识别" / 输出为空
        """
        prompts = []
        empty_mask = []
        for text in texts:
            if not text.strip():
                prompts.append(None)
                empty_mask.append(True)
            else:
                prompts.append(self._build_prompt(text))
                empty_mask.append(False)

        non_empty_indices = [i for i, empty in enumerate(empty_mask) if not empty]
        non_empty_prompts = [prompts[i] for i in non_empty_indices]

        # 默认：空输入 → ("", False)
        results: List[Tuple[str, bool]] = [("", False)] * len(texts)

        if non_empty_prompts:
            outputs = self._llm.generate(non_empty_prompts, self._sampling_params)
            for idx, output in zip(non_empty_indices, outputs):
                text_out = output.outputs[0].text.strip()
                formatted = format_numbered_output(text_out)
                # 判断是否有效输出
                is_valid = bool(formatted) and formatted != "无法识别"
                results[idx] = (formatted, is_valid)

        return results


# =============================================================================
# 主处理流程
# =============================================================================

def process_year(
    year: str,
    config: Config,
    processor: OfflineBatchProcessor,
    logger: logging.Logger,
):
    """处理单个年份的数据"""
    input_path = config.input_path(year)
    output_dir = config.output_parquet_dir(year)
    checkpoint = CheckpointManager(config.checkpoint_path(year), logger)

    if checkpoint.is_complete:
        logger.info(f"年份 {year} 已完成，跳过")
        return

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    os.makedirs(output_dir, exist_ok=True)

    stats = ProcessingStats()

    # 统计总行数（用 pandas 解析，正确处理含换行符的引号字段）
    logger.info(f"统计 {year} 年数据量...")
    total_rows = sum(
        len(chunk) for chunk in pd.read_csv(
            input_path, encoding="utf-8-sig",
            chunksize=50000, usecols=[0], on_bad_lines="skip",
        )
    )
    stats.total = total_rows
    logger.info(f"  总行数: {total_rows:,}")

    rows_done = checkpoint.rows_done
    chunks_written = checkpoint.chunks_written
    if rows_done > 0:
        logger.info(f"  从断点恢复: 已完成 {rows_done:,} 行")

    pbar = tqdm(total=total_rows, initial=rows_done, desc=f"Step2 {year}",
                unit="条", ncols=100) if tqdm else None

    # vLLM 推理批次大小与写盘 chunk 解耦：
    #   llm_batch_size  — 每次调用 llm.generate() 的条数，直接决定 GPU 的 batch size
    #                     设大一些（如 50000）让 vLLM 一次看到足够多请求，调度更高效
    #   write_chunk_size — 每次写盘的条数，影响断点粒度和内存峰值，可独立设置
    llm_batch_size = config.llm_batch_size

    # 读取 CSV（pandas 仍按 write_chunk_size 分块，仅用于流式读取控制内存）
    # 注意：skiprows=range(N) 会构建 N 元素集合（~360MB），且某些 pandas 版本与
    # chunksize 组合有 bug 导致读取 0 行。改用 skiprows=int 跳过前 N 行，更可靠。
    col_names = pd.read_csv(
        input_path, encoding="utf-8-sig", nrows=0
    ).columns.tolist()

    if rows_done > 0:
        csv_iter = pd.read_csv(
            input_path,
            encoding="utf-8-sig",
            chunksize=config.write_chunk_size,
            skiprows=rows_done + 1,  # 跳过 header(1行) + 已处理行(rows_done行)
            header=None,
            names=col_names,
            on_bad_lines="skip",
        )
    else:
        csv_iter = pd.read_csv(
            input_path,
            encoding="utf-8-sig",
            chunksize=config.write_chunk_size,
            on_bad_lines="skip",
        )

    # 滚动缓冲：积累到 llm_batch_size 条后一次性推理，再按 write_chunk_size 写盘
    pending_dfs: List[pd.DataFrame] = []
    pending_rows: int = 0

    def flush_pending():
        """把 pending_dfs 中的数据一次性推理并写盘"""
        nonlocal pending_rows, chunks_written, rows_done

        if not pending_dfs:
            return

        big_df = pd.concat(pending_dfs, ignore_index=True)
        pending_dfs.clear()
        pending_rows = 0

        if config.text_column not in big_df.columns:
            raise ValueError(
                f"CSV 缺少字段 '{config.text_column}'，"
                f"实际列: {list(big_df.columns)}"
            )

        texts = big_df[config.text_column].fillna("").astype(str).tolist()

        # 一次性推理（GPU 连续满负荷运行，无写盘等待间隙）
        batch_results = processor.process_batch(texts)

        big_df[config.output_column] = [r[0] for r in batch_results]
        big_df["year"] = str(year)  # 强制 string，避免 PyArrow 推断为 dictionary/int

        # 按 write_chunk_size 分批写盘
        for write_start in range(0, len(big_df), config.write_chunk_size):
            write_df = big_df.iloc[write_start: write_start + config.write_chunk_size]
            chunk_file = Path(output_dir) / f"chunk_{chunks_written:06d}.parquet"
            table = pa.Table.from_pandas(write_df, preserve_index=False)
            # 将所有 dictionary 列转为基础类型，保证 schema 一致
            table = _normalize_schema(table)
            pq_writer = pq.ParquetWriter(str(chunk_file), table.schema, compression="snappy")
            pq_writer.write_table(table)
            pq_writer.close()
            chunks_written += 1

        n = len(big_df)
        rows_done += n
        stats.processed += n
        stats.success += sum(1 for r in batch_results if r[1])
        stats.failed += sum(1 for r in batch_results if not r[1])

        checkpoint.update(rows_done, chunks_written)

        if pbar:
            pbar.update(n)
            pbar.set_postfix_str(f"速度: {stats.speed:.0f} 条/s")

        logger.info(f"  batch 完成 {n} 条 | {stats}")

    for csv_chunk_df in csv_iter:
        csv_chunk_df = csv_chunk_df.reset_index(drop=True)
        pending_dfs.append(csv_chunk_df)
        pending_rows += len(csv_chunk_df)

        if pending_rows >= llm_batch_size:
            flush_pending()

    # 处理尾部不足 llm_batch_size 的剩余数据
    flush_pending()

    if pbar:
        pbar.close()

    # 合并所有 chunk 文件
    _merge_chunks(output_dir, chunks_written, logger)
    checkpoint.mark_complete()

    logger.info(
        f"年份 {year} 完成: "
        f"成功 {stats.success:,} | 失败 {stats.failed:,} | "
        f"耗时 {stats.elapsed:.0f}s | 速度 {stats.speed:.1f} 条/s"
    )


def _normalize_schema(table: pa.Table) -> pa.Table:
    """将 dictionary 编码列转为基础类型，确保跨 chunk schema 一致。"""
    new_arrays = []
    new_fields = []
    for i, field in enumerate(table.schema):
        col = table.column(i)
        if pa.types.is_dictionary(field.type):
            col = col.cast(field.type.value_type)
            field = pa.field(field.name, field.type.value_type)
        new_arrays.append(col)
        new_fields.append(field)
    return pa.table(new_arrays, schema=pa.schema(new_fields))


def _merge_chunks(output_dir: str, chunks_written: int, logger: logging.Logger):
    """将所有 chunk_*.parquet 合并为 data.parquet，自动处理 schema 不一致。"""
    chunk_files = sorted(Path(output_dir).glob("chunk_*.parquet"))
    if not chunk_files:
        logger.warning(f"未找到 chunk 文件: {output_dir}")
        return

    logger.info(f"合并 {len(chunk_files)} 个 chunk 文件...")
    output_file = Path(output_dir) / "data.parquet"

    # 以第一个文件为基准 schema（已 normalize）
    # 注意：pq.read_table() 内部使用 Dataset API，会扫描目录内所有 parquet 文件
    # 并尝试合并 schema，导致 ArrowTypeError。改用 ParquetFile 只读单个文件。
    first_table = _normalize_schema(pq.ParquetFile(str(chunk_files[0])).read())
    writer = pq.ParquetWriter(str(output_file), first_table.schema, compression="snappy")
    writer.write_table(first_table)

    for cf in chunk_files[1:]:
        table = _normalize_schema(pq.ParquetFile(str(cf)).read())
        # 按基准 schema cast，防止 string/large_string 等细微差异
        table = table.cast(first_table.schema)
        writer.write_table(table)

    writer.close()

    for cf in chunk_files:
        cf.unlink()

    logger.info(f"✓ 合并完成: {output_file}")


# =============================================================================
# 命令行入口
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="职位描述标准化 - 云端 vLLM Offline Batch 版")
    parser.add_argument("-c", "--config", required=True, help="配置文件路径")
    parser.add_argument("--year", required=True, help="处理年份，如 2021")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config.from_yaml(args.config)
    year = args.year
    logger = setup_logging(year)

    logger.info("=" * 60)
    logger.info(f"云端 Step 2 | 年份: {year} | 模式: vLLM Offline Batch")
    logger.info(f"模型: {config.llm_model}")
    logger.info(f"输入: {config.input_path(year)}")
    logger.info(f"输出: {config.output_parquet_dir(year)}")
    logger.info("=" * 60)

    extractor = DutyExtractor(config)
    processor = OfflineBatchProcessor(config, extractor, logger)
    processor.load_model()

    process_year(year, config, processor, logger)


if __name__ == "__main__":
    main()
