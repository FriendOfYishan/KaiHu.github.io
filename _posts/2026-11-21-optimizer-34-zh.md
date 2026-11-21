---
layout: research-post
title: "34 · 倒数残差与稳定 34"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 34
permalink: /research/zh/34/
alternate_url: /research/en/34/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

33 暴露了权重失配和相减不稳定。

## 这一步改进什么？

用倒数残差表达二阶估计，区分稳定实现与 PL 条件快率。

## 结果与局限：为什么没达到目标？

常数版筛查有梯度越界失败；合法种子上与匹配平方版本几乎相同，没有普遍优势。

## 下一个方向是什么？

明确适用任务，再检验记忆分离。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/34.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/34/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/34/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/34/core.py" | relative_url }})

[← 33]({{ "/research/zh/33/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [34-count →]({{ "/research/zh/34-count/" | relative_url }})
