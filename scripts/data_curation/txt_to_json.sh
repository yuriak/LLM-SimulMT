#!/bin/bash

SRC_FILE="$1"
TGT_FILE="$2"
OUTPUT_FILE="$3"

python3 <<EOF
import os, json, re
merged = []
with open("$SRC_FILE", "r", encoding="utf-8") as fs, \
        open("$TGT_FILE", "r", encoding="utf-8") as ft:
    src_lines = fs.readlines()
    tgt_lines = ft.readlines()

if len(src_lines) != len(tgt_lines):
    print(f"[Error] Line mismatch in {src_path} & {tgt_path}")
    print(f"  src: {len(src_lines)}, tgt: {len(tgt_lines)}")
    raise SystemExit(1)

for s, t in zip(src_lines, tgt_lines):
    merged.append({
        "src": s.strip(),
        "tgt": t.strip()
    })

with open("$OUTPUT_FILE", "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False)

echo "Done. Total pairs: {len(merged)}"
echo "Saved to $OUTPUT_FILE"
EOF