---
layout: research-post
title: "44 · 先归一化再积累动量"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 44
permalink: /research/zh/44/
alternate_url: /research/en/44/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

43 暴露同预算优化不足与当前分母偏差。

## 这一步改进什么？

分离统计预测器和更新动量，使用历史残差尺度。

## 结果与局限：为什么没达到目标？

44 未超过 43/AdamW；预设尺度上界截住大量坐标，分离结构未带来可测收益。

## 下一个方向是什么？

减少信号在分母中的自我压制，并检验位移预算。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/44.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/44/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/44/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/44/core.py" | relative_url }})

[← 43]({{ "/research/zh/43/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [45 →]({{ "/research/zh/45/" | relative_url }})
