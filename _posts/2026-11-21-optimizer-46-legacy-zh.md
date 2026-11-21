---
layout: research-post
title: "46-legacy · 旧 46a/46b：先修公式再看结果"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 46-legacy
permalink: /research/zh/46-legacy/
alternate_url: /research/en/46-legacy/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

45 的短程领先需要更简洁的结构解释。

## 这一步改进什么？

修复旧公式的梯度位置、残差时序、自指统计与尺度重锚定。

## 结果与局限：为什么没达到目标？

修复后 46a 明显落后，46b 也未建立优势；它们不同于后来的成对截断 46。

## 下一个方向是什么？

冻结旧结果，单独筛查新的残差计时机制。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/46-legacy.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/46-legacy/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/46-legacy/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/46-legacy/core.py" | relative_url }})

[← 46]({{ "/research/zh/46/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [47 →]({{ "/research/zh/47/" | relative_url }})
