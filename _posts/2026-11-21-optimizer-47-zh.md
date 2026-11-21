---
layout: research-post
title: "47 · 投影删掉噪声，也删掉方向"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 47
permalink: /research/zh/47/
alternate_url: /research/en/47/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

46 的门槛失败促使我们测试结构噪声。

## 这一步改进什么？

组内中心化或冻结符号投影，配合残差超阈计数。

## 结果与局限：为什么没达到目标？

四任务失败对应未校准减均值版本，不能直接归罪于所有冻结投影；固定阈值还出现计数饱和或近乎停用。

## 下一个方向是什么？

保留全部方向，用残差行列统计恢复逐坐标尺度。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/47.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/47/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/47/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/47/core.py" | relative_url }})

[← 46-legacy]({{ "/research/zh/46-legacy/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [48 →]({{ "/research/zh/48/" | relative_url }})
