# -*- coding: utf-8 -*-
"""构建 Applied Soft Computing (elsarticle) 格式 PDF。

流程：
1. 同步最新 11_实验章节.md 的 §4 到 20_论文全文_插图版.md；
2. 从合稿提取正文（§1 起，跳过摘要工作页），剥去标题编号，生成 body.md；
3. 调 pandoc 转 LaTeX，后处理（caption* 去 LaTeX 自编号、剥 figs/ 前缀、删 label）。
"""
import re
import subprocess
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent  # paper 目录
FULL = PAPER / "20_论文全文_插图版.md"
EXP = PAPER / "11_实验章节.md"
LATEX = PAPER / "latex"

CH4 = "# 4 实验（Experiments）"


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8-sig")


def sync_ch4():
    full_lines = read(FULL).splitlines()
    exp_lines = read(EXP).splitlines()
    start = full_lines.index(CH4)
    end = next(i for i in range(start + 1, len(full_lines))
               if full_lines[i].startswith("# "))
    new = full_lines[:start] + exp_lines + full_lines[end:]
    FULL.write_text("\n".join(new) + "\n", encoding="utf-8")
    print(f"ch4 synced: {start}..{end} -> {len(exp_lines)} lines "
          f"(total {len(new)})")


def make_body_md():
    full_lines = read(FULL).splitlines()
    start = next(i for i, l in enumerate(full_lines)
                 if l.startswith("# 1 "))
    body = full_lines[start:]
    out = []
    pat = re.compile(r"^(#{1,6})\s+\d+(\.\d+)*\s+")
    for l in body:
        m = pat.match(l)
        out.append(pat.sub(r"\1 ", l) if m else l)
    (LATEX / "body.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"body.md: {len(out)} lines from line {start + 1}")


def postprocess():
    tex = (LATEX / "body.tex").read_text(encoding="utf-8")
    # 图 caption 改无编号（正文引用手动 Fig.N，避免双重编号）
    n_cap = tex.count("\\caption{")
    tex = tex.replace("\\caption{", "\\caption*{")
    # 删 pandoc 自动 label（正文不用 \ref）
    tex, n_lab = re.subn(r"(?m)^\\label\{[^}]*\}\s*$\n?", "", tex)
    # 图路径改由 graphicspath 解析
    tex = tex.replace("{figs/", "{")
    (LATEX / "body.tex").write_text(tex, encoding="utf-8")
    print(f"postprocess: {n_cap} captions -> caption*, {n_lab} labels removed")


def pandoc():
    cmd = ["pandoc", "body.md", "-f", "markdown", "-t", "latex",
           "--top-level-division=section", "-o", "body.tex"]
    subprocess.run(cmd, cwd=LATEX, check=True)
    print("pandoc done")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    LATEX.mkdir(exist_ok=True)
    if step in ("sync", "all"):
        sync_ch4()
    if step in ("body", "all"):
        make_body_md()
    if step in ("pandoc", "all"):
        pandoc()
    if step in ("post", "all"):
        postprocess()
