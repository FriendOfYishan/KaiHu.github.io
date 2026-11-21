# 45 · 残差分母：短期领先为何反转 / Residual denominators over longer budgets

这是优化器核心与最小合成入口或 GPU 核摘录；完整历史训练表还需要原数据、模型和训练配方，本精简附件不含这些大材料。 / Optimizer mechanism or GPU excerpt only; historical training metrics require the original data, model and recipe. 合成入口需要 CUDA + Triton，本次本地仅做语法与依赖静态核对，没有重跑 GPU 实验。 / The CUDA/Triton entry received static checks only locally; no GPU experiment was rerun.


## 最短入口 / Minimal entry

```bash
python reproduce.py
```

依赖 / Dependencies: Python 3.10+; 按 core.py 导入安装 numpy、scipy、sympy、matplotlib、torch；Triton 核需要兼容 CUDA GPU 与 triton。原环境版本以 report.md 为准。 / Install only the numerical packages imported by core.py; Triton excerpts require a compatible CUDA GPU. See report.md for historical environments.

`core.py` 保存原文件或按 AST 提取的优化器依赖闭包；不包含 SSH、服务器控制、密钥、数据集或检查点。`SOURCE_MANIFEST.json` 记录原文件哈希和摘录方式。新写的小反例会标成 new-audit。 / The manifest distinguishes original files, dependency slices and new audit aids.

`reproduce.py`（若有）是为本次归档新增的最小机制入口，不是当年完整训练脚本；其 docstring 说明检查内容。 / When present, reproduce.py is a new minimal mechanism entry, not the full historical training script.

## 证据 / Evidence

短预算领先没有持续：97.9M 参数、328.5M token 上验证 NLL 比 AdamW 差 0.0184；双 BF16 慢 EMA 另有累加误差。

Long-budget NLL lost by 0.0184; BF16 slow-EMA accumulation added error.

`report.md` 是已有汇报的原文快照（若无独立汇报则明确注明），原报告内的本地相对路径仅是历史引用；请按本目录文件表阅读。中文报告未冒充为英文译本。 / Reports remain in their original language; the blog supplies a short English account. Historical relative paths inside source snapshots may refer to omitted material.
