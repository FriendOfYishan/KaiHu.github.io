---
layout: research-post
title: "37-layer · Layer37：状态减半，信息也少了"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 37-layer
permalink: /research/zh/37-layer/
alternate_url: /research/en/37-layer/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

37 尝试用耗散约束尺度刷新。

## 这一步改进什么？

把逐坐标尺度压成每层一个标量，只留一份完整动量。

## 结果与局限：为什么没达到目标？

状态约减半，但到 NLL=6 的时间慢约 52.9%，峰值显存只少约 3.13%；预算消融没有质量收益。

## 下一个方向是什么？

补回低成本的残差空间信息。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/37-layer.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/37-layer/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/37-layer/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/37-layer/core.py" | relative_url }})

[← 37-batch]({{ "/research/zh/37-batch/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [38 →]({{ "/research/zh/38/" | relative_url }})
