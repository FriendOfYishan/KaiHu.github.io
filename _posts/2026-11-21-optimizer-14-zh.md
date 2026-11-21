---
layout: research-post
title: "14 · AWA：简化公式，也暴露障碍"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 14
permalink: /research/zh/14/
alternate_url: /research/en/14/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

13 提供了状态补偿与时间加权差分。

## 这一步改进什么？

纠正首步和权重，将二阶矩简化，并区分普通差分与创新差分。

## 结果与局限：为什么没达到目标？

原差分有有限位移障碍；创新差分的 PL 快率属于预条件 SGD；GPU 实验未胜调参 AdamW。

## 下一个方向是什么？

检查前移混合和历史偏差能否真正改善。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/14.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/14/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/14/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/14/core.py" | relative_url }})

[← 13]({{ "/research/zh/13/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [15 →]({{ "/research/zh/15/" | relative_url }})
