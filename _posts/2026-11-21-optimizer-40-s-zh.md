---
layout: research-post
title: "40-s · 40-S：删除信号门控"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 40-s
permalink: /research/zh/40-s/
alternate_url: /research/en/40-s/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

40 的受控收益没有迁移到语言模型。

## 这一步改进什么？

删除 K 和变换输入，让动量继续平均原始梯度。

## 结果与局限：为什么没达到目标？

这是工程简化与消融，不是已经证明的新理论贡献；需分别看图像和语言报告。

## 下一个方向是什么？

若再分配增益，先保留未识别的残差信号。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/40-s.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/40-s/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/40-s/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/40-s/core.py" | relative_url }})

[← 40]({{ "/research/zh/40/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [41 →]({{ "/research/zh/41/" | relative_url }})
