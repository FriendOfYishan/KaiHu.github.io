---
layout: research-post
title: "21 · 局部可识别的联合选择"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 21
permalink: /research/zh/21/
alternate_url: /research/en/21/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

20 后，短轨迹指数仍难迁移。

## 这一步改进什么？

在固定状态上联合优化条件跟踪风险和预期下降。

## 结果与局限：为什么没达到目标？

chain 中指数预测偏离网格最优，下降证书不为正；新规则仍依赖可估计统计量。

## 下一个方向是什么？

用同一势函数合并下降与未来记忆代价。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/21.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/21/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/21/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/21/core.py" | relative_url }})

[← 20]({{ "/research/zh/20/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [22 →]({{ "/research/zh/22/" | relative_url }})
