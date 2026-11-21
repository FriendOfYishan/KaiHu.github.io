---
layout: research-post
title: "26 · 不识别漂移也能闭合能量界"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 26
permalink: /research/zh/26/
alternate_url: /research/en/26/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

25 的低成本在线识别未通过跨任务检验。

## 这一步改进什么？

用原二阶递推吸收预条件器能量变化，不增加梯度或向量状态。

## 结果与局限：为什么没达到目标？

闭合了条件非凸下降界，但主导阶数没有超过 18，也未证明优于 AdamW。

## 下一个方向是什么？

检查跟踪快率与优化快率之间的区别。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/26.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/26/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/26/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/26/core.py" | relative_url }})

[← 25]({{ "/research/zh/25/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [27 →]({{ "/research/zh/27/" | relative_url }})
