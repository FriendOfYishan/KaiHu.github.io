---
layout: research-post
title: "20 · 从跟踪误差转向有效更新"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 20
permalink: /research/zh/20/
alternate_url: /research/en/20/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

19 的扫描不能单靠渐近幂律解释。

## 这一步改进什么？

分离首批权重、偏差修正、分母几何和跨时间相关项。

## 结果与局限：为什么没达到目标？

旧实验未保存完整方向与预条件器，不能倒推出对齐已获验证。

## 下一个方向是什么？

在同一固定状态上研究可识别的下一步收益。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/20.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/20/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/20/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/20/core.py" | relative_url }})

[← 19]({{ "/research/zh/19/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [21 →]({{ "/research/zh/21/" | relative_url }})
