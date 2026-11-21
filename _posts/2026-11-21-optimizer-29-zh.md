---
layout: research-post
title: "29 · 按实际移动分配逐坐标记忆"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 29
permalink: /research/zh/29/
alternate_url: /research/en/29/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

28 强调修正应针对真实反馈。

## 这一步改进什么？

让一二阶共享逐坐标权重，把移动代理接入记忆分配。

## 结果与局限：为什么没达到目标？

理论代理改善没有兑现语言模型表现：后续 FineWeb 报告中该候选明显落后 AdamW。

## 下一个方向是什么？

保留有效的一阶 EMA，重新审视二阶统计。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/29.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/29/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/29/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/29/core.py" | relative_url }})

[← 28]({{ "/research/zh/28/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [30 →]({{ "/research/zh/30/" | relative_url }})
