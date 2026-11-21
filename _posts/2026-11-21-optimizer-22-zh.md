---
layout: research-post
title: "22 · 把移动与记忆放进同一势函数"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 22
permalink: /research/zh/22/
alternate_url: /research/en/22/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

21 的局部准则还需连接未来误差。

## 这一步改进什么？

反向预计算标量日程，将当前下降与未来跟踪误差统一。

## 结果与局限：为什么没达到目标？

证书改善只针对同一状态；需要曲率上界和 O(T) 标量预计算，不代表跨轨迹胜出。

## 下一个方向是什么？

更直接处理有符号的对齐项。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/22.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/22/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/22/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/22/core.py" | relative_url }})

[← 21]({{ "/research/zh/21/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [23 →]({{ "/research/zh/23/" | relative_url }})
