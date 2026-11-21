---
layout: research-post
title: "51-fused · 51 融合：质量结论之后再谈工程"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 51-fused
permalink: /research/zh/51-fused/
alternate_url: /research/en/51-fused/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

51 短预算有收益，但纯 torch 的墙钟明显更慢。

## 这一步改进什么？

提供可读简式、Triton 融合及单遍矩阵读写，复核数值一致性。

## 结果与局限：为什么没达到目标？

503 步训练 26.40s 对 fused AdamW 24.08s，状态 4.012 对 8 B/参数；后段调参没有改善长预算终点。FineWeb 目录仅有试跑记录，不能写成完整 300M 对照已完成。

## 下一个方向是什么？

完成 FineWeb 主实验，以同质量墙钟和峰值显存做裁决。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/51-fused.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/51-fused/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/51-fused/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/51-fused/core.py" | relative_url }})

[← 51]({{ "/research/zh/51/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }})
