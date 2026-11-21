---
layout: research-post
title: "08 · The predictive-shrink branch"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: en
entry_id: 08
permalink: /research/en/08/
alternate_url: /research/zh/08/
---

Retrospective compiled on 2026-11-21. Original identifiers are preserved; adjacent numbers need not form a direct lineage.

## What came before?

Follow the controller bottleneck.

## What changed?

Couple shrinkage with predictive sampling.

## What did not work, or remains limited?

Short MNIST gains came with higher time and update tails.

## What comes next?

Return to gradients at the current parameters.

## Small attachment and reproduction scope

[ZIP]({{ "/_assets/optimizer-journey/08.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/08/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/08/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/08/core.py" | relative_url }})

[← 07]({{ "/research/en/07/" | relative_url }}) · [指引 / Index]({{ "/research/en/" | relative_url }}) · [09 →]({{ "/research/en/09/" | relative_url }})
