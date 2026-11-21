# 33 · 少一次乘法，多一个失配项 / Ordinary harmonic averaging

运行现有代数/机制核验，不复现训练表。 / Runs algebra or mechanism checks, not historical training tables.


## 最短入口 / Minimal entry

```bash
python core.py
```

依赖 / Dependencies: Python 3.10+; 按 core.py 导入安装 numpy、scipy、sympy、matplotlib、torch；Triton 核需要兼容 CUDA GPU 与 triton。原环境版本以 report.md 为准。 / Install only the numerical packages imported by core.py; Triton excerpts require a compatible CUDA GPU. See report.md for historical environments.

`core.py` 保存原文件或按 AST 提取的优化器依赖闭包；不包含 SSH、服务器控制、密钥、数据集或检查点。`SOURCE_MANIFEST.json` 记录原文件哈希和摘录方式。新写的小反例会标成 new-audit。 / The manifest distinguishes original files, dependency slices and new audit aids.

`reproduce.py`（若有）是为本次归档新增的最小机制入口，不是当年完整训练脚本；其 docstring 说明检查内容。 / When present, reproduce.py is a new minimal mechanism entry, not the full historical training script.

## 证据 / Evidence

一阶与调和统计的历史权重失配，需要非负保护和额外误差项。

Mismatched history weights introduce extra error and positivity protection.

`report.md` 是已有汇报的原文快照（若无独立汇报则明确注明），原报告内的本地相对路径仅是历史引用；请按本目录文件表阅读。中文报告未冒充为英文译本。 / Reports remain in their original language; the blog supplies a short English account. Historical relative paths inside source snapshots may refer to omitted material.
