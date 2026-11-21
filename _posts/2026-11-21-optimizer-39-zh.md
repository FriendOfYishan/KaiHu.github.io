---
layout: research-post
title: "39 · 重构得更准，参考方向却不够好"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 39
permalink: /research/zh/39/
alternate_url: /research/en/39/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

38 提出只压缩残差。

## 这一步改进什么？

证明结构化误差界，构造一状态信号重构并做冻结快照比较。

## 结果与局限：为什么没达到目标？

局部代理误差明显降低，但 Full39-exact 本身落后强 AdamW；逼近它不能保证训练变好。

## 下一个方向是什么？

把重构误差接到真实下降响应。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/39.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/39/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/39/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/39/core.py" | relative_url }})

[← 38-shape]({{ "/research/zh/38-shape/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [39-b →]({{ "/research/zh/39-b/" | relative_url }})
