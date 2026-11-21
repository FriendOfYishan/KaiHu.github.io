---
layout: research-post
title: "27 · 跟踪快了，优化可能停了"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 27
permalink: /research/zh/27/
alternate_url: /research/en/27/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

26 已得到闭合下降分析。

## 这一步改进什么？

把更快衰减日程同时代入跟踪界与完整优化界。

## 结果与局限：为什么没达到目标？

参数几乎不动也能跟踪准确；可求和步长可能造成优化停滞。

## 下一个方向是什么？

修正估计器的漂移，而非仅加快步长衰减。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/27.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/27/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/27/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/27/core.py" | relative_url }})

[← 26]({{ "/research/zh/26/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [28 →]({{ "/research/zh/28/" | relative_url }})
