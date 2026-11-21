---
layout: research-post
title: "09 · Auditing multilevel shrinkage"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: en
entry_id: 09
permalink: /research/en/09/
alternate_url: /research/zh/09/
---

Retrospective compiled on 2026-11-21. Original identifiers are preserved; adjacent numbers need not form a direct lineage.

## What came before?

Return to current-point gradients.

## What changed?

Correct the EMA transformation and audit displacement noise.

## What did not work, or remains limited?

The proposed equivalence fails; joint performance goals were unmet.

## What comes next?

Treat cumulative updates as a distinct design.

## Small attachment and reproduction scope

[ZIP]({{ "/_assets/optimizer-journey/09.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/09/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/09/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/09/core.py" | relative_url }})

[← 08]({{ "/research/en/08/" | relative_url }}) · [指引 / Index]({{ "/research/en/" | relative_url }}) · [10 →]({{ "/research/en/10/" | relative_url }})
