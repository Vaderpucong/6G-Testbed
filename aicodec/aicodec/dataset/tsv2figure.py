#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从官方 MMBench TSV 文件一键生成本仓库使用的图片 + manifest.jsonl。

MMBench 官方分发的是单个 TSV 文件（每行一条样本，图片以 base64 编码存
在某一列里），而 One-D-Piece <-> Qwen3-VL 的测试管线 (aitestbed/scenarios
/odp_image_udp.py, run_odp_mmbench_pipeline.py) 消费的是"解压好的 jpg +
manifest.jsonl"目录。本脚本负责这一步转换。

加固要点（相对最初的 pandas 版本）：
- 纯标准库 csv 解析，不依赖 pandas，避免 NaN/dtype 相关的隐藏坑。
- image 列里出现的跨行引用（值是另一行的 index）会被解析到不动点，
  并检测自引用环/断链。
- 每个 base64 payload 解码后都用 PIL 校验，损坏数据直接报错而不是
  静默写出空文件。
- 幂等：已存在且校验通过的图片会被跳过，只有损坏/缺失的才会重新解码。

用法:
    python aicodec/aicodec/dataset/tsv2figure.py \
        --tsv /path/to/MMBench_DEV_EN.tsv \
        --out-dir aicodec/aicodec/dataset/MMBench/MMBench_DEV_EN \
        --split dev

输出:
    <out-dir>/<index>.jpg (或 .png)
    <out-dir>/manifest.jsonl
    (可选 --report) <report>.json -- 失败项/缺失项/孤立文件汇总
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import sys
from pathlib import Path

from PIL import Image

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2 ** 31 - 1)

CHOICE_COLS = ("A", "B", "C", "D")


def _norm_col(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def load_rows(tsv_path: Path) -> list[dict]:
    with open(tsv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames:
            reader.fieldnames = [_norm_col(c) for c in reader.fieldnames]
        rows = []
        for r in reader:
            idx = (r.get("index") or "").strip()
            if not idx:
                continue
            rows.append({k: (v or "") for k, v in r.items()})
    return rows


def resolve_references(image_map: dict) -> tuple[dict, list]:
    """Resolve numeric cross-row image references to a fixed point."""
    resolved = dict(image_map)
    for _ in range(len(resolved) + 2):
        changed = False
        for k, v in list(resolved.items()):
            v = v.strip()
            if v.isdigit():
                tgt = resolved.get(v)
                if tgt is None:
                    continue
                if not tgt.strip().isdigit():
                    resolved[k] = tgt
                    changed = True
        if not changed:
            break
    problems = []
    for k, v in resolved.items():
        v = v.strip()
        if v.isdigit():
            kind = "broken_target" if v not in image_map else "cycle"
            problems.append({"index": k, "ref": v, "kind": kind})
    return resolved, problems


def _guess_ext(raw: bytes) -> str:
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    return "jpg"


def verify_payload(payload: str) -> bytes:
    raw = base64.b64decode(payload)
    with Image.open(io.BytesIO(raw)) as im:
        im.verify()
    return raw


def file_is_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def convert(tsv_path: Path, out_dir: Path, default_split: str,
            limit, overwrite_images: bool, report_path=None) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    rows = load_rows(tsv_path)
    required = {"index", "question", "a", "b", "image"}
    if rows:
        missing_cols = required - set(rows[0].keys())
        if missing_cols:
            raise SystemExit(
                f"TSV missing required columns {sorted(missing_cols)}; "
                f"found: {list(rows[0].keys())}")

    image_map = {r["index"].strip(): r.get("image", "") for r in rows}
    resolved, ref_problems = resolve_references(image_map)
    n_refs = sum(1 for v in image_map.values() if v.strip().isdigit())

    written = skipped_valid = failed = 0
    failures = []
    with manifest_path.open("w", encoding="utf-8") as fout:
        count = 0
        for r in rows:
            if limit is not None and count >= limit:
                break
            idx = r["index"].strip()
            payload = resolved.get(idx, "").strip()
            if not payload:
                failed += 1
                failures.append({"index": idx, "error": "empty_image_cell"})
                continue

            existing = next(iter(out_dir.glob(f"{idx}.*")), None)
            if existing and not overwrite_images and file_is_valid(existing):
                skipped_valid += 1
                img_name = existing.name
            else:
                try:
                    raw = verify_payload(payload)
                except Exception as e:  # noqa: BLE001
                    failed += 1
                    failures.append({
                        "index": idx,
                        "error": f"{type(e).__name__}: {e}",
                    })
                    continue
                ext = _guess_ext(raw)
                img_name = f"{idx}.{ext}"
                img_path = out_dir / img_name
                tmp_path = img_path.with_suffix(img_path.suffix + ".part")
                tmp_path.write_bytes(raw)
                tmp_path.replace(img_path)
                written += 1

            choices = {c: r.get(c.lower(), "").strip() for c in CHOICE_COLS}
            choices = {c: v for c, v in choices.items() if v}
            entry = {
                "index": idx,
                "image": img_name,
                "question": r.get("question", "").strip(),
                "hint": r.get("hint", "").strip(),
                "choices": choices,
                "answer": r.get("answer", "").strip(),
                "category": r.get("category", "").strip(),
                "l2_category": r.get("l2_category", "").strip(),
                "split": r.get("split", "").strip() or default_split,
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1

    disk = {p.stem for p in out_dir.glob("*")
            if p.suffix in (".jpg", ".jpeg", ".png")}
    missing = sorted(set(image_map) - disk,
                      key=lambda x: int(x) if x.isdigit() else 0)
    orphan = sorted(disk - set(image_map),
                     key=lambda x: int(x) if x.isdigit() else 0)

    print(f"[done] wrote/verified {count} rows to {manifest_path} "
          f"(images: {out_dir})")
    print(f"written={written} skipped_valid={skipped_valid} failed={failed} "
          f"cross_row_refs={n_refs} ref_problems={len(ref_problems)}")
    if missing:
        print(f"  missing_on_disk={len(missing)} sample={missing[:20]}",
              file=sys.stderr)
    if orphan:
        print(f"  orphan_files={len(orphan)} sample={orphan[:20]}",
              file=sys.stderr)

    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({
            "tsv": str(tsv_path), "rows": len(rows), "written": written,
            "skipped_valid": skipped_valid, "failed": failed,
            "failures": failures, "cross_row_refs": n_refs,
            "ref_problems": ref_problems, "missing_on_disk": missing,
            "orphan_files": orphan,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    if failed or ref_problems:
        print("WARNING: decode finished with problems, see failed/"
              "ref_problems above.", file=sys.stderr)
    return count


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tsv", required=True, type=Path,
                    help="official MMBench TSV file path")
    p.add_argument("--out-dir", required=True, type=Path,
                    help="output dir, e.g. aicodec/aicodec/dataset/MMBench/"
                         "MMBench_DEV_EN")
    p.add_argument("--split", default="dev",
                    help="default split value if TSV has no split column "
                         "(default: dev)")
    p.add_argument("--limit", type=int, default=None,
                    help="convert only the first N rows, for a quick smoke "
                         "test (default: all)")
    p.add_argument("--overwrite-images", action="store_true",
                    help="force re-decode even if a valid image with the "
                         "same name already exists")
    p.add_argument("--report", type=Path, default=None,
                    help="optional: write a decode report json (failures/"
                         "missing/orphans)")
    args = p.parse_args()
    convert(args.tsv, args.out_dir, args.split, args.limit,
            args.overwrite_images, args.report)


if __name__ == "__main__":
    main()