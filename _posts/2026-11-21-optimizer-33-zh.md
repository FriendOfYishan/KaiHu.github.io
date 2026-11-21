---
layout: research-post
title: "33 · 少一次乘法，多一个失配项"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 33
permalink: /research/zh/33/
alternate_url: /research/en/33/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

32 仍保留调和遗忘与权重状态。

## 这一步改进什么？

用普通调和平均及预设总权重简化递推。

## 结果与局限：为什么没达到目标？

一阶与调和统计的历史权重失配，需要非负保护和额外误差项。

## 下一个方向是什么？

用倒数残差恒等式避免差分消减。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/33.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/33/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/33/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/33/core.py" | relative_url }})

[← 32]({{ "/research/zh/32/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [34 →]({{ "/research/zh/34/" | relative_url }})
