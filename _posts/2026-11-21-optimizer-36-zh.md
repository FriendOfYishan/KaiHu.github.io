---
layout: research-post
title: "36 · Flash：核快不等于训练快"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 36
permalink: /research/zh/36/
alternate_url: /research/en/36/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

35 提醒我们同时核对计算预算。

## 这一步改进什么？

用运动误差与噪声代价决定旧动量何时释放，检查批量实现。

## 结果与局限：为什么没达到目标？

局部核计时更低，完整训练未明显加速，在线控制器还会增加开销。

## 下一个方向是什么？

用动量耗散直接约束尺度刷新。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/36.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/36/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/36/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/36/core.py" | relative_url }})

[← 35]({{ "/research/zh/35/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [37 →]({{ "/research/zh/37/" | relative_url }})
