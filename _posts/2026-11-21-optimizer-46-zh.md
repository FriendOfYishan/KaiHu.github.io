---
layout: research-post
title: "46 · 成对截断的残差时钟"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 46
permalink: /research/zh/46/
alternate_url: /research/en/46/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

42/44 与旧 46 提示统计量和方向应分开审计。

## 这一步改进什么？

用旧动量、旧尺度和成对截断残差构造新的低状态候选。

## 结果与局限：为什么没达到目标？

新三阶段筛选中只完成前两阶段；46 虽胜 42，但落后 Layer37/AdamW，未过门槛，取消 30M 阶段。

## 下一个方向是什么？

先做投影后的信号与噪声诊断，再决定训练。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/46.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/46/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/46/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/46/core.py" | relative_url }})

[← 45]({{ "/research/zh/45/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [46-legacy →]({{ "/research/zh/46-legacy/" | relative_url }})
