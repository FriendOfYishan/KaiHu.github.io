---
layout: research-post
title: "12 · 曲率—噪声杠杆的反例"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: zh
entry_id: 12
permalink: /research/zh/12/
alternate_url: /research/en/12/
---

这是截至 2026-11-21 的回顾整理，编号沿用原研究材料；相邻编号不一定是直接继承关系。

## 前一步是什么？

11 的快率依赖已知曲率。

## 这一步改进什么？

核对峰度识别曲率和指数分离噪声的提案，写出反例。

## 结果与局限：为什么没达到目标？

峰度不能一般识别曲率，指数变换也不能唯一分离信号与噪声。

## 下一个方向是什么？

只用已有矩构造有明确范围的补偿系数。

## 精简附件与复现范围

[ZIP]({{ "/_assets/optimizer-journey/12.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/12/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/12/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/12/core.py" | relative_url }})

[← 11]({{ "/research/zh/11/" | relative_url }}) · [指引 / Index]({{ "/research/zh/" | relative_url }}) · [13 →]({{ "/research/zh/13/" | relative_url }})
