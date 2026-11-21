---
layout: research-post
title: "40 · Allocating update noise"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: en
entry_id: 40
permalink: /research/en/40/
alternate_url: /research/zh/40/
---

Retrospective compiled on 2026-11-21. Original identifiers are preserved; adjacent numbers need not form a direct lineage.

## What came before?

Follow the weak-reference diagnosis.

## What changed?

Allocate update noise using compressed residuals.

## What did not work, or remains limited?

Controlled-model gains did not transfer to WikiText-2.

## What comes next?

Remove gating and retain raw-gradient momentum.

## Small attachment and reproduction scope

[ZIP]({{ "/_assets/optimizer-journey/40.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/40/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/40/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/40/core.py" | relative_url }})

[← 39-b]({{ "/research/en/39-b/" | relative_url }}) · [指引 / Index]({{ "/research/en/" | relative_url }}) · [40-s →]({{ "/research/en/40-s/" | relative_url }})
