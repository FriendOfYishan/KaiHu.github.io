---
layout: research-post
title: "16 · 历史偏差抵消不等于无代价降噪"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 16
permalink: /research/zh/16/
alternate_url: /research/en/16/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

15 要求先厘清滤波器定义。

## 这一步改进什么？

统一正权重族，研究双平均、趋势校准及指数和对数权重。

## 结果与局限：为什么没达到目标？

减小滞后通常增加噪声或状态；滤波恒等式不构成神经网络收敛结论。

## 下一个方向是什么？

建立漂移、噪声和记忆的可计算模型。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/16.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/16/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/16/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/16/core.py" | relative_url }})

[← 15]({{ "/research/zh/15/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [17 →]({{ "/research/zh/17/" | relative_url }})
