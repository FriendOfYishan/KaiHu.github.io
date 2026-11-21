---
layout: research-post
title: "40 · 从恢复分母转向分配噪声"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 40
permalink: /research/zh/40/
alternate_url: /research/en/40/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

39/39-B 说明准确恢复某个方向还不够。

## 这一步改进什么？

用压缩残差决定更新噪声分配，给出条件非凸稳定分析。

## 结果与局限：为什么没达到目标？

合成结构与受控目标上的成功未迁移到 WikiText-2；信号判定可能过于激进。

## 下一个方向是什么？

删除门控，保留原始梯度动量作为消融。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/40.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/40/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/40/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/40/core.py" | relative_url }})

[← 39-b]({{ "/research/zh/39-b/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [40-s →]({{ "/research/zh/40-s/" | relative_url }})
