---
layout: research-post
title: "48 · 一份动量加行列残差"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 48
permalink: /research/zh/48/
alternate_url: /research/en/48/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

47 的硬投影限制了可移动方向。

## 这一步改进什么？

恢复全部方向，以动量平方加行列残差重建逐坐标预条件。

## 结果与局限：为什么没达到目标？

四任务状态约减半、质量接近但语言仍落后 AdamW；未融合实现更慢。

## 下一个方向是什么？

检验信号加权尺度和 RMS 保护。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/48.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/48/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/48/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/48/core.py" | relative_url }})

[← 47]({{ "/research/zh/47/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [49 →]({{ "/research/zh/49/" | relative_url }})
