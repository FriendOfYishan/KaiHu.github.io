---
layout: research-post
title: "44 · Predictable residual momentum"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: en
entry_id: 44
permalink: /research/en/44/
alternate_url: /research/zh/44/
---

Retrospective compiled on 2026-11-21. Original identifiers are preserved; adjacent numbers need not form a direct lineage.

## What came before?

Follow the current-scale bias concern.

## What changed?

Normalize with historical residuals before accumulating momentum.

## What did not work, or remains limited?

The scale cap suppressed many coordinates; no quality win appeared.

## What comes next?

Revisit signal suppression and displacement limits.

## Small attachment and reproduction scope

[ZIP]({{ "/_assets/optimizer-journey/44.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/44/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/44/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/44/core.py" | relative_url }})

[← 43]({{ "/research/en/43/" | relative_url }}) · [指引 / Index]({{ "/research/en/" | relative_url }}) · [45 →]({{ "/research/en/45/" | relative_url }})
