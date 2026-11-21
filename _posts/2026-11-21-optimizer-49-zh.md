---
layout: research-post
title: "49 · 放大步子未必改善方向"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 49
permalink: /research/zh/49/
alternate_url: /research/en/49/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

48 的低状态结构已经基本站住。

## 这一步改进什么？

减弱信号对分母的占用，加入张量 RMS 保护。

## 结果与局限：为什么没达到目标？

同一四任务 harness 的语言结果比 48 更差；步幅放大不能替代方向改善。

## 下一个方向是什么？

寻找行列压缩误差可精确抵消的增益。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/49.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/49/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/49/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/49/core.py" | relative_url }})

[← 48]({{ "/research/zh/48/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [50 →]({{ "/research/zh/50/" | relative_url }})
