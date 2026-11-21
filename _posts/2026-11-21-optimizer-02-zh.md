---
layout: research-post
title: "02 · CrossAdamW 的条件快率"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 02
permalink: /research/zh/02/
alternate_url: /research/en/02/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

01 说明历史项必须按实际时间顺序分析。

## 这一步改进什么？

使用延迟、有界分母，将动量新信息权重与步长匹配。

## 结果与局限：为什么没达到目标？

平均梯度平方快率依赖额外消失噪声条件，并非交叉项自动带来；没有训练胜出证据。

## 下一个方向是什么？

先核对既有实验到底支持哪些结论。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/02.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/02/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/02/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/02/core.py" | relative_url }})

[← 01]({{ "/research/zh/01/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [03 →]({{ "/research/zh/03/" | relative_url }})
