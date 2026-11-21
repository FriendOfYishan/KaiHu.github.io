---
layout: research-post
title: "43 · 压缩历史仍没解决优化不足"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 43
permalink: /research/zh/43/
alternate_url: /research/en/43/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

42 的累计块尺度可能过于粗糙。

## 这一步改进什么？

使用当前逐坐标残差和压缩历史块状态，做语言与图像对照。

## 结果与局限：为什么没达到目标？

语言训练 probe 与验证均落后 AdamW，首先是优化不足；当前分母还可能引入方向偏差。

## 下一个方向是什么？

用历史残差定尺度，再积累归一化动量。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/43.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/43/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/43/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/43/core.py" | relative_url }})

[← 42]({{ "/research/zh/42/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [44 →]({{ "/research/zh/44/" | relative_url }})
