---
layout: research-post
title: "45 · 残差分母：短期领先为何反转"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 45
permalink: /research/zh/45/
alternate_url: /research/en/45/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

44 的位移幅度与短程质量一起下降。

## 这一步改进什么？

从分母扣除动量可解释部分，允许可靠方向走到截断上限。

## 结果与局限：为什么没达到目标？

短预算领先没有持续：97.9M 参数、328.5M token 上验证 NLL 比 AdamW 差 0.0184；双 BF16 慢 EMA 另有累加误差。

## 下一个方向是什么？

恢复 FP32 统计，寻找少状态而不损伤方向的结构。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/45.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/45/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/45/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/45/core.py" | relative_url }})

[← 44]({{ "/research/zh/44/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [46 →]({{ "/research/zh/46/" | relative_url }})
