---
layout: research-post
title: "09 · 多层缩减：先纠正变量变换"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 09
permalink: /research/zh/09/
alternate_url: /research/en/09/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

08 后回到当前参数求梯度。

## 这一步改进什么？

修正去除 EMA 的代数变换，计算完整时间相关方差并测试位移缩减。

## 结果与局限：为什么没达到目标？

原累加等价式不成立；新递推也未同时达到更快、更稳、更省的目标。

## 下一个方向是什么？

把累加作为独立设计，重建相容矩。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/09.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/09/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/09/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/09/core.py" | relative_url }})

[← 08]({{ "/research/zh/08/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [10 →]({{ "/research/zh/10/" | relative_url }})
