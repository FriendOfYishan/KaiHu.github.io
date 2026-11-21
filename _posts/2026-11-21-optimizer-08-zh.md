---
layout: research-post
title: "08 · 一个系数的前移梯度分支"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 08
permalink: /research/zh/08/
alternate_url: /research/en/08/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

07 的控制器需要难以估计的统计量。

## 这一步改进什么？

用同一系数耦合平方差缩减与前移采样，一次反传、两份状态。

## 结果与局限：为什么没达到目标？

MNIST 短实验训练 loss 改善，但耗时和更新尾部更高；后续研究约束转回当前位置梯度。

## 下一个方向是什么？

直接研究参数递推，不再依赖前移采样。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/08.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/08/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/08/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/08/core.py" | relative_url }})

[← 07]({{ "/research/zh/07/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [09 →]({{ "/research/zh/09/" | relative_url }})
