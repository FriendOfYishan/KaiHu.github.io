---
layout: research-post
title: "06 · 校准 v3：平滑尺度仍不够"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 06
permalink: /research/zh/06/
alternate_url: /research/en/06/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

早期交叉项只在局部统计意义上有吸引力。

## 这一步改进什么？

加入历史噪声校准、有效学习率变化约束及参考实现。

## 结果与局限：为什么没达到目标？

扩大等额学习率搜索后未胜 AdamW，完整更新尾部反而更大。

## 下一个方向是什么？

用真实下降预算判断是否收缩。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/06.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/06/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/06/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/06/core.py" | relative_url }})

[← 05]({{ "/research/zh/05/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [07 →]({{ "/research/zh/07/" | relative_url }})
