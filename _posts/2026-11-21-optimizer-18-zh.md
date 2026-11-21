---
layout: research-post
title: "18 · 记忆与移动速度的联合界"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 18
permalink: /research/zh/18/
alternate_url: /research/en/18/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

17 的外生模型不能完整描述训练反馈。

## 这一步改进什么？

把参数移动导致的梯度漂移纳入误差递推，联合选择记忆和步长。

## 结果与局限：为什么没达到目标？

实验证明的是固定任务上的交互；一般最优性和泛化收益仍未建立。

## 下一个方向是什么？

直接测量同一参数位置的跟踪误差。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/18.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/18/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/18/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/18/core.py" | relative_url }})

[← 17]({{ "/research/zh/17/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [19 →]({{ "/research/zh/19/" | relative_url }})
