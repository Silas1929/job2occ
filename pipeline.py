#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
云端 Pipeline 总调度

功能：
- 按年份串行或多卡并行运行 Step1 → Step2 → Step3 → Step4
- Step2/Step3 均使用 vLLM offline batch 模式，每个子进程独立加载模型
- 支持从指定 Step 断点续跑
- 步骤间行数校验，检测静默丢数据

架构说明：
    pipeline.py 仅负责调度——按年份顺序/并行启动子进程。
    每个步骤脚本在自身进程内加载 vLLM 模型，完成后释放 GPU 显存。
    无需在 pipeline 层面管理 vLLM 服务生命周期。

目录结构（上云后）：
    $PIPELINE_BASE_DIR/          默认 /root/autodl-tmp
    ├── job2occ/                 本脚本所在目录
    ├── models/
    ├── data/raw/                原始 CSV（Step1 输入）
    ├── data/input/              Step1 清洗后的 CSV（Step2 输入）
    ├── data/reference/          大典 xlsx + O*NET 参考表
    └── data/output/             Pipeline 输出

使用方式：
    # 单年测试
    python pipeline.py --year 2021

    # 全部年份顺序执行（单 GPU）
    python pipeline.py --all-years

    # 多 GPU 并行（需设置 NUM_GPUS 环境变量）
    NUM_GPUS=4 python pipeline.py --all-years

    # 从 Step3 开始断点续跑
    python pipeline.py --year 2021 --start-from step3

    # 只跑某一个 Step
    python pipeline.py --year 2021 --only step2
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

# =============================================================================
# 常量
# =============================================================================

YEARS = list(range(2015, 2025))   # 2015-2024

BASE_DIR = Path(__file__).parent
PIPELINE_BASE_DIR = os.environ.get("PIPELINE_BASE_DIR", "/root/autodl-tmp")

STEP_SCRIPTS = {
    "step1": BASE_DIR / "step1_clean" / "step1_clean_content.py",
    "step2": BASE_DIR / "step2_vllm" / "standardize_vllm.py",
    "step3": BASE_DIR / "step3_embedding" / "embed_and_match_gpu.py",
    "step4": BASE_DIR / "step4_onet" / "add_onet_mapping.py",
}

STEP_CONFIGS = {
    "step1": BASE_DIR / "step1_clean" / "config.yaml",
    "step2": BASE_DIR / "step2_vllm" / "config.yaml",
    "step3": BASE_DIR / "step3_embedding" / "config.yaml",
    "step4": BASE_DIR / "step4_onet" / "config.yaml",
}

STEP_ORDER = ["step1", "step2", "step3", "step4"]

# 步骤间输出路径模板（用于行数校验）
STEP_OUTPUT_PATHS = {
    "step1": "{base}/data/input/recruitment_infos_content_{year}_step1_clean.csv",
    "step2": "{base}/data/output/standardized/year={year}/data.parquet",
    "step3": "{base}/data/output/matched/year={year}/data.parquet",
    # step4 输出为 Hive 分区目录，不做精确行数校验
}


# =============================================================================
# 日志
# =============================================================================

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("job2occ")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)

    # 日志文件持久化
    log_dir = Path(PIPELINE_BASE_DIR) / "data" / "output" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(
        log_dir / f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log",
        encoding="utf-8",
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


logger = setup_logger()


# =============================================================================
# 行数校验
# =============================================================================

def _count_rows(path: str) -> Optional[int]:
    """统计文件行数。支持 CSV 和 Parquet。"""
    p = Path(path)
    if not p.exists():
        return None
    try:
        if p.suffix == ".csv":
            import csv
            with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
                reader = csv.reader(f)
                next(reader, None)  # 跳过 header
                return sum(1 for _ in reader)
        elif p.suffix == ".parquet":
            import pyarrow.parquet as pq
            return pq.ParquetFile(path).metadata.num_rows
    except Exception:
        pass
    return None


def validate_step_output(step: str, year: int) -> Optional[int]:
    """校验步骤输出，返回行数（如果可计算）。"""
    template = STEP_OUTPUT_PATHS.get(step)
    if not template:
        return None
    path = template.format(base=PIPELINE_BASE_DIR, year=year)
    return _count_rows(path)


def check_row_counts(prev_step: str, curr_step: str, year: int):
    """比较相邻步骤的输出行数，不一致时警告。"""
    prev_rows = validate_step_output(prev_step, year)
    curr_rows = validate_step_output(curr_step, year)

    if prev_rows is not None and curr_rows is not None:
        if prev_rows != curr_rows:
            logger.warning(
                f"行数校验: {prev_step}→{curr_step} year={year} | "
                f"{prev_step} 输出 {prev_rows:,} 行, {curr_step} 输出 {curr_rows:,} 行 "
                f"(差异 {abs(prev_rows - curr_rows):,} 行)"
            )
        else:
            logger.info(
                f"行数校验: {prev_step}→{curr_step} year={year} | "
                f"一致 ({curr_rows:,} 行)"
            )


# =============================================================================
# 单步执行
# =============================================================================

def run_step(step: str, year: int, gpu_idx: int,
             max_retries: int = 10, retry_delay: int = 30) -> bool:
    """
    以子进程方式运行单个 Step，返回是否成功。
    每个步骤脚本自行初始化 vLLM 模型，完成后自动释放。

    自动重试：vLLM 在长时间运行时可能因 CUDA 内存错误崩溃（已知 bug）。
    断点续传保证每次重启从上次写盘位置继续，最多损失 llm_batch_size 条。
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

    cmd = [
        sys.executable,
        str(STEP_SCRIPTS[step]),
        "--config", str(STEP_CONFIGS[step]),
        "--year", str(year),
    ]

    for attempt in range(1, max_retries + 1):
        logger.info(f"  执行 (尝试 {attempt}/{max_retries}): {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            env=env,
            cwd=str(STEP_SCRIPTS[step].parent),
        )
        if result.returncode == 0:
            return True

        if attempt < max_retries:
            logger.warning(
                f"  {step} year={year} 失败 (returncode={result.returncode})，"
                f"{retry_delay}s 后自动重试 ({attempt}/{max_retries})..."
            )
            time.sleep(retry_delay)
        else:
            logger.error(f"  {step} year={year} 达到最大重试次数 {max_retries}，放弃")

    return False


# =============================================================================
# 单年处理
# =============================================================================

def run_year(year: int, gpu_idx: int, steps: List[str]) -> bool:
    """顺序执行指定年份的所有 Step"""
    logger.info(f"\n{'='*50}")
    logger.info(f"开始处理年份: {year}（GPU {gpu_idx}）")
    logger.info(f"执行步骤: {steps}")
    logger.info(f"{'='*50}")

    start_time = time.time()

    for i, step in enumerate(steps):
        logger.info(f"\n--- {step.upper()} | 年份 {year} ---")
        ok = run_step(step, year, gpu_idx)
        if not ok:
            logger.error(f"FAILED: {step} for year {year}")
            return False

        # 步骤间行数校验
        if i > 0:
            check_row_counts(steps[i - 1], step, year)

    elapsed = time.time() - start_time
    logger.info(f"\n✓ 年份 {year} 全部完成，耗时 {elapsed/3600:.1f} 小时")
    return True


# =============================================================================
# 命令行参数解析
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="云端招聘数据 Pipeline 调度器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 单年测试
  python pipeline.py --year 2021

  # 全部年份（单 GPU）
  python pipeline.py --all-years

  # 多 GPU 并行（4 张卡）
  NUM_GPUS=4 python pipeline.py --all-years

  # 从 step3 开始（step1/step2 已完成）
  python pipeline.py --year 2021 --start-from step3

  # 只跑 step2
  python pipeline.py --year 2021 --only step2

  # 指定 GPU 编号
  python pipeline.py --year 2021 --gpu 1
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--year", type=int, choices=YEARS, help="处理单个年份")
    group.add_argument("--all-years", action="store_true", help="处理全部年份")

    parser.add_argument(
        "--start-from",
        choices=STEP_ORDER,
        default="step1",
        help="从指定步骤开始（断点续跑）",
    )
    parser.add_argument(
        "--only",
        choices=STEP_ORDER,
        help="只运行指定步骤",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="指定使用的 GPU 编号（单年模式）",
    )
    return parser.parse_args()


# =============================================================================
# 主函数
# =============================================================================

def main():
    args = parse_args()

    # 确定执行步骤列表
    if args.only:
        steps = [args.only]
    else:
        start_idx = STEP_ORDER.index(args.start_from)
        steps = STEP_ORDER[start_idx:]

    num_gpus = int(os.environ.get("NUM_GPUS", "1"))
    logger.info(f"云端 Pipeline 启动")
    logger.info(f"执行步骤: {steps}")
    logger.info(f"可用 GPU 数: {num_gpus}")
    logger.info(f"PIPELINE_BASE_DIR: {PIPELINE_BASE_DIR}")

    # -------------------------------------------------------------------------
    # 单年模式
    # -------------------------------------------------------------------------
    if args.year:
        run_year(args.year, args.gpu, steps)
        return

    # -------------------------------------------------------------------------
    # 全年模式
    # -------------------------------------------------------------------------
    if num_gpus == 1:
        # 单 GPU：顺序处理各年份
        for year in YEARS:
            ok = run_year(year, gpu_idx=0, steps=steps)
            if not ok:
                logger.error(f"年份 {year} 失败，停止后续处理")
                sys.exit(1)

    else:
        # 多 GPU：每张卡处理不同年份，子进程自行管理 vLLM 加载
        # 使用 ThreadPoolExecutor（而非 ProcessPoolExecutor）：
        # 实际 GPU 工作在 subprocess.run 子进程内完成，
        # 线程池仅负责并发调度 subprocess，不涉及 CUDA context。
        # ThreadPoolExecutor 更轻量，且避免 fork 导致的 CUDA 问题。
        logger.info(f"多 GPU 并行模式：{num_gpus} 张卡，按年份分配")

        def run_year_on_gpu(year_gpu: tuple) -> tuple:
            year, gpu_idx = year_gpu
            ok = run_year(year, gpu_idx, steps)
            return year, ok

        year_gpu_pairs = [(year, i % num_gpus) for i, year in enumerate(YEARS)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as pool:
            futures = {pool.submit(run_year_on_gpu, pair): pair[0] for pair in year_gpu_pairs}
            failed_years = []
            for future in concurrent.futures.as_completed(futures):
                year = futures[future]
                try:
                    _, ok = future.result()
                    if not ok:
                        failed_years.append(year)
                        logger.error(f"年份 {year} 处理失败")
                except Exception as e:
                    failed_years.append(year)
                    logger.error(f"年份 {year} 异常: {e}")

        if failed_years:
            logger.error(f"以下年份处理失败: {sorted(failed_years)}")
            sys.exit(1)

    logger.info("\n✓ Pipeline 全部完成")


if __name__ == "__main__":
    main()
