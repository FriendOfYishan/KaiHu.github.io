---
layout: research-post
title: "01 · 把 v2.1 写成可检验的理论"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 01
permalink: /research/zh/01/
alternate_url: /research/en/01/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

从 Shrink AdamW 的历史交叉项出发。

## 这一步改进什么？

补上实际 EMA 历史与非平稳漂移项，明确条件风险。

## 结果与局限：为什么没达到目标？

固定点降噪不等于移动参数下的训练收益；原有非平稳缺口不能忽略。

## 下一个方向是什么？

把采样前可知的分母与记忆日程写进完整算法。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/01.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/01/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/01/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/01/core.py" | relative_url }})

[指引 / Index]({{ "/research/zh/" | relative_url }}) · [02 →]({{ "/research/zh/02/" | relative_url }})
