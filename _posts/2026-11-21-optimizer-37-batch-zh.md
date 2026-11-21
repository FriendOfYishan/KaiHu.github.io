---
layout: research-post
title: "37-batch · Layer37-B：更快步数有梯度代价"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 37-batch
permalink: /research/zh/37-batch/
alternate_url: /research/en/37-batch/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

Layer37 丢失了坐标尺度。

## 这一步改进什么？

在保持低状态的同时缓慢增加有效批量，改善特定步数界。

## 结果与局限：为什么没达到目标？

更多样本梯度要计入预算；减噪无法恢复已经压掉的坐标信息。

## 下一个方向是什么？

同 token、同墙钟核对质量，而非只数更新步。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/37-batch.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/37-batch/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/37-batch/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/37-batch/core.py" | relative_url }})

[← 37]({{ "/research/zh/37/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [37-layer →]({{ "/research/zh/37-layer/" | relative_url }})
