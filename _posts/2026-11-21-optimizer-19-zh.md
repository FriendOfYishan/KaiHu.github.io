---
layout: research-post
title: "19 · 有限步指数扫描的陷阱"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 19
permalink: /research/zh/19/
alternate_url: /research/en/19/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

18 给出了渐近的记忆—步长关系。

## 这一步改进什么？

在移动参数前测完整训练梯度，检查有限步最佳指数。

## 结果与局限：为什么没达到目标？

启动项与有限时间常数会改变最佳指数；诊断全梯度也不是免费预算。

## 下一个方向是什么？

加入启动遗忘、归一化几何与时间相关分析。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/19.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/19/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/19/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/19/core.py" | relative_url }})

[← 18]({{ "/research/zh/18/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [20 →]({{ "/research/zh/20/" | relative_url }})
