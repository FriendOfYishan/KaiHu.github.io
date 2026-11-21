---
layout: research-post
title: "42 · 残差时钟：把统计对齐计算块"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 42
permalink: /research/zh/42/
alternate_url: /research/en/42/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

41 的增益分配没有兑现收益。

## 这一步改进什么？

只累积不可预测残差，原始 EMA 配合每 1024 参数一个统计量。

## 结果与局限：为什么没达到目标？

块压缩仍丢失坐标尺度，累计分母在持续噪声下可能增长；快筛未建立竞争优势。

## 下一个方向是什么？

测试更细的当前残差尺度和压缩历史。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/42.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/42/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/42/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/42/core.py" | relative_url }})

[← 41]({{ "/research/zh/41/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [43 →]({{ "/research/zh/43/" | relative_url }})
