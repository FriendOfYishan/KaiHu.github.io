---
layout: research-post
title: "34-count · 计数恢复：换一个有结构的问题"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 34-count
permalink: /research/zh/34-count/
alternate_url: /research/en/34-count/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

34 的通用训练优势尚未兑现。

## 这一步改进什么？

在独立 Poisson、已校准基线及持续上移模型下构造常数状态检测恢复。

## 结果与局限：为什么没达到目标？

同预算实验虽快于低风险 EMA，却没有超过 CUSUM＋重启均值；模型内阶最优不等于常数或实际恢复表现最佳，也不能转记到原 34。

## 下一个方向是什么？

检验分布失配、变化幅度与校准误差。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/34-count.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/34-count/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/34-count/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/34-count/core.py" | relative_url }})

[← 34]({{ "/research/zh/34/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [34-task →]({{ "/research/zh/34-task/" | relative_url }})
