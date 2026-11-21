---
layout: research-post
title: "51 · 历史与创新分开：进展与边界"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 51
permalink: /research/zh/51/
alternate_url: /research/en/51/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

48 的历史方向稳健，50 的预算恒等式缺乏控制力度。

## 这一步改进什么？

历史项逐坐标归一化，当前创新走平方可加增益通道并截断到 ±2。

## 结果与局限：为什么没达到目标？

短预算三种子 NLL 5.4718 对 AdamW 5.5313；40M 调参后最佳值 4.6874 对 4.6842，未建立最终质量优势；融合后仍未全面更快。

## 下一个方向是什么？

保留截断消融结论，完成新语料长预算配对与同质量墙钟核验。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/51.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/51/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/51/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/51/core.py" | relative_url }})

[← 50]({{ "/research/zh/50/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [51-fused →]({{ "/research/zh/51-fused/" | relative_url }})
