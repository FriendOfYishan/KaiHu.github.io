---
layout: research-post
title: "10 · 相容累计矩与完整位移噪声"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 10
permalink: /research/zh/10/
alternate_url: /research/en/10/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

09 表明累加不能冒充 EMA 的等价写法。

## 这一步改进什么？

构造保正累计矩，用归一化累计状态之差更新参数。

## 结果与局限：为什么没达到目标？

得到特定条件下的保证，但短闭环实验不支持全面优于调参 AdamW；属于自适应对偶平均路线。

## 下一个方向是什么？

在可解二次模型里追问尾部速率。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/10.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/10/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/10/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/10/core.py" | relative_url }})

[← 09]({{ "/research/zh/09/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [11 →]({{ "/research/zh/11/" | relative_url }})
