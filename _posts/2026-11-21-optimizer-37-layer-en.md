---
layout: research-post
title: "37-layer · Layer37: the cost of compression"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: en
entry_id: 37-layer
permalink: /research/en/37-layer/
alternate_url: /research/zh/37-layer/
---

Retrospective compiled on 2026-11-21. Original identifiers are preserved; adjacent numbers need not form a direct lineage.

## What came before?

Follow dissipation-paid scales.

## What changed?

Store one scalar scale per layer.

## What did not work, or remains limited?

State savings did not yield matched-quality speed or large peak-memory savings.

## What comes next?

Restore inexpensive spatial residual information.

## Small attachment and reproduction scope

[ZIP]({{ "/_assets/optimizer-journey/37-layer.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/37-layer/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/37-layer/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/37-layer/core.py" | relative_url }})

[← 37-batch]({{ "/research/en/37-batch/" | relative_url }}) · [指引 / Index]({{ "/research/en/" | relative_url }}) · [38 →]({{ "/research/en/38/" | relative_url }})
