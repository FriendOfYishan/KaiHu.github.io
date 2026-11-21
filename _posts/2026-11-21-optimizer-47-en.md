---
layout: research-post
title: "47 · Projection can remove useful directions"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: en
entry_id: 47
permalink: /research/en/47/
alternate_url: /research/zh/47/
---

Retrospective compiled on 2026-11-21. Original identifiers are preserved; adjacent numbers need not form a direct lineage.

## What came before?

Follow the pair-clipping failure.

## What changed?

Project group directions and count threshold crossings.

## What did not work, or remains limited?

The uncalibrated centering variant failed; calibrated frozen projections are distinct.

## What comes next?

Keep all directions and compress only residual statistics.

## Small attachment and reproduction scope

[ZIP]({{ "/_assets/optimizer-journey/47.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/47/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/47/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/47/core.py" | relative_url }})

[← 46-legacy]({{ "/research/en/46-legacy/" | relative_url }}) · [指引 / Index]({{ "/research/en/" | relative_url }}) · [48 →]({{ "/research/en/48/" | relative_url }})
