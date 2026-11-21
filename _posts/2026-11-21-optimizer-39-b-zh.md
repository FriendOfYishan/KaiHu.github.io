---
layout: research-post
title: "39-b · 把重构接回真实下降"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 39-b
permalink: /research/zh/39-b/
alternate_url: /research/en/39-b/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

39 的完整参考本身训练不足。

## 这一步改进什么？

用下降响应分析替代仅比较重构代理误差。

## 结果与局限：为什么没达到目标？

条件下降改善不等于真实训练轨迹胜出；参考方向和压缩误差必须同时检查。

## 下一个方向是什么？

把核心目标改为信号—噪声的有效分配。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/39-b.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/39-b/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/39-b/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/39-b/core.py" | relative_url }})

[← 39]({{ "/research/zh/39/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [40 →]({{ "/research/zh/40/" | relative_url }})
