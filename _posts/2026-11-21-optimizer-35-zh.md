---
layout: research-post
title: "35 · 一阶 EMA、二阶长记忆"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 35
permalink: /research/zh/35/
alternate_url: /research/en/35/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

34 之后回看共同日程的矩消融。

## 这一步改进什么？

保留常数一阶系数，研究时间衰减二阶及非 PL 方差缩减。

## 结果与局限：为什么没达到目标？

二阶 AWA 的小收益会随学习率变号；更快步数界不能隐去额外梯度调用。

## 下一个方向是什么？

回到一次梯度预算，研究证据释放动量。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/35.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/35/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/35/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/35/core.py" | relative_url }})

[← 34-task]({{ "/research/zh/34-task/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [36 →]({{ "/research/zh/36/" | relative_url }})
