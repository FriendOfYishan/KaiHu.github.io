---
layout: research-post
title: "38 · 保留信号，只压缩残差"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 38
permalink: /research/zh/38/
alternate_url: /research/en/38/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

Layer37 的状态减半没有带来同质量时间优势。

## 这一步改进什么？

保留逐坐标动量，用残差行列统计恢复少量各向异性。

## 结果与局限：为什么没达到目标？

行列重建不能保留所有坐标结构；几何近似本身不保证真实下降。

## 下一个方向是什么？

定义残差压缩的优化误差，再测试参考方向。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/38.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/38/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/38/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/38/core.py" | relative_url }})

[← 37-layer]({{ "/research/zh/37-layer/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [38-shape →]({{ "/research/zh/38-shape/" | relative_url }})
