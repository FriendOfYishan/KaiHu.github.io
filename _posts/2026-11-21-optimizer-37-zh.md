---
layout: research-post
title: "37 · 让尺度刷新支付耗散代价"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 37
permalink: /research/zh/37/
alternate_url: /research/en/37/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

36 的在线证据控制成本偏高。

## 这一步改进什么？

把尺度变化纳入动量耗散预算，保持低状态递推。

## 结果与局限：为什么没达到目标？

定理和合成检查不足以确认 Transformer 质量；预算的独立收益仍需消融。

## 下一个方向是什么？

将尺度压到每层一个标量，实测信息损失。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/37.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/37/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/37/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/37/core.py" | relative_url }})

[← 36]({{ "/research/zh/36/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [37-batch →]({{ "/research/zh/37-batch/" | relative_url }})
