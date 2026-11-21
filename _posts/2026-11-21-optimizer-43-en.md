---
layout: research-post
title: "43 · Compressed history and current scales"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: en
entry_id: 43
permalink: /research/en/43/
alternate_url: /research/zh/43/
---

Retrospective compiled on 2026-11-21. Original identifiers are preserved; adjacent numbers need not form a direct lineage.

## What came before?

Follow the coarse residual clock.

## What changed?

Combine current coordinate residuals with compressed history.

## What did not work, or remains limited?

Both training-probe and validation NLL trailed AdamW.

## What comes next?

Scale predictably before accumulating update momentum.

## Small attachment and reproduction scope

[ZIP]({{ "/_assets/optimizer-journey/43.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/43/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/43/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/43/core.py" | relative_url }})

[← 42]({{ "/research/en/42/" | relative_url }}) · [指引 / Index]({{ "/research/en/" | relative_url }}) · [44 →]({{ "/research/en/44/" | relative_url }})
