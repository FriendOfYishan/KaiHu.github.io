---
layout: research-post
title: "38-shape · 形状预算的平行分支"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 38-shape
permalink: /research/zh/38-shape/
alternate_url: /research/en/38-shape/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

Layer37 后在另一工作目录继续探索形状信息。

## 这一步改进什么？

用形状位预算和受控尺度带宽改善条件常数，并检验 sign/量化形状的分配效应。

## 结果与局限：为什么没达到目标？

受控稀有 token 实验只复现了原质量差距的一小部分，不足以确认唯一归因；该 38 与残差行列 38 定义不同，不能混用结果。

## 下一个方向是什么？

分别保存来源，再用同一 harness 对照。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/38-shape.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/38-shape/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/38-shape/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/38-shape/core.py" | relative_url }})

[← 38]({{ "/research/zh/38/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [39 →]({{ "/research/zh/39/" | relative_url }})
