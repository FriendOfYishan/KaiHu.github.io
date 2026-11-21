---
layout: research-post
title: "15 · AWA 前移混合：原式先过合法性"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 15
permalink: /research/zh/15/
alternate_url: /research/en/15/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

14 暴露了时间权重与差分问题。

## 这一步改进什么？

审计总步数 T 下的混合权重，分别定义可实现修复。

## 结果与局限：为什么没达到目标？

原权重在训练期大于 1，二阶量可为负、末步可除零；同批双点也需要额外梯度成本。

## 下一个方向是什么？

研究历史滤波偏差的抵消，而非隐藏代价。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/15.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/15/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/15/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/15/core.py" | relative_url }})

[← 14]({{ "/research/zh/14/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [16 →]({{ "/research/zh/16/" | relative_url }})
