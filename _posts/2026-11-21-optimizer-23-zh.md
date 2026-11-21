---
layout: research-post
title: "23 · 不加梯度的对齐能量分析"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 23
permalink: /research/zh/23/
alternate_url: /research/en/23/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

22 的范数界可能丢掉方向信息。

## 这一步改进什么？

保留有符号分解，利用动量能量和同样本协方差。

## 结果与局限：为什么没达到目标？

比粗界更细不等于实际下降更好；预条件器能量变化项仍需控制。

## 下一个方向是什么？

在可达固定状态上比较界的松紧。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/23.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/23/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/23/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/23/core.py" | relative_url }})

[← 22]({{ "/research/zh/22/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [24 →]({{ "/research/zh/24/" | relative_url }})
