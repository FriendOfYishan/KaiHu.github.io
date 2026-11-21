---
layout: research-post
title: "40-s · 40-S without signal gating"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: en
entry_id: 40-s
permalink: /research/en/40-s/
alternate_url: /research/zh/40-s/
---

Retrospective compiled on 2026-11-21. Original identifiers are preserved; adjacent numbers need not form a direct lineage.

## What came before?

Follow candidate 40's transfer failure.

## What changed?

Remove signal gating and retain raw-gradient momentum.

## What did not work, or remains limited?

This is a simplification, not a demonstrated new theoretical contribution.

## What comes next?

Preserve ambiguous residual signal in any later allocation.

## Small attachment and reproduction scope

[ZIP]({{ "/_assets/optimizer-journey/40-s.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/40-s/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/40-s/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/40-s/core.py" | relative_url }})

[← 40]({{ "/research/en/40/" | relative_url }}) · [指引 / Index]({{ "/research/en/" | relative_url }}) · [41 →]({{ "/research/en/41/" | relative_url }})
