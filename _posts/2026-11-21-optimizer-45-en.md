---
layout: research-post
title: "45 · Residual denominators over longer budgets"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: en
entry_id: 45
permalink: /research/en/45/
alternate_url: /research/zh/45/
---

Retrospective compiled on 2026-11-21. Original identifiers are preserved; adjacent numbers need not form a direct lineage.

## What came before?

Follow the restricted-step diagnosis.

## What changed?

Remove explained signal from the denominator and cap larger updates.

## What did not work, or remains limited?

Long-budget NLL lost by 0.0184; BF16 slow-EMA accumulation added error.

## What comes next?

Preserve accurate statistics and improve low-state geometry.

## Small attachment and reproduction scope

[ZIP]({{ "/_assets/optimizer-journey/45.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/45/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/45/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/45/core.py" | relative_url }})

[← 44]({{ "/research/en/44/" | relative_url }}) · [指引 / Index]({{ "/research/en/" | relative_url }}) · [46 →]({{ "/research/en/46/" | relative_url }})
