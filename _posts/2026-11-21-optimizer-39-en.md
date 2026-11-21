---
layout: research-post
title: "39 · Reconstructing the wrong reference"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: en
entry_id: 39
permalink: /research/en/39/
alternate_url: /research/zh/39/
---

Retrospective compiled on 2026-11-21. Original identifiers are preserved; adjacent numbers need not form a direct lineage.

## What came before?

Follow residual compression.

## What changed?

Bound reconstruction error and test frozen snapshots.

## What did not work, or remains limited?

The exact reference itself underperformed AdamW.

## What comes next?

Relate reconstruction to actual descent response.

## Small attachment and reproduction scope

[ZIP]({{ "/_assets/optimizer-journey/39.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/39/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/39/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/39/core.py" | relative_url }})

[← 38-shape]({{ "/research/en/38-shape/" | relative_url }}) · [指引 / Index]({{ "/research/en/" | relative_url }}) · [39-b →]({{ "/research/en/39-b/" | relative_url }})
