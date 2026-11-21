---
layout: research-post
title: "34-count · Count-calibrated recovery"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: en
entry_id: 34-count
permalink: /research/en/34-count/
alternate_url: /research/zh/34-count/
---

Retrospective compiled on 2026-11-21. Original identifiers are preserved; adjacent numbers need not form a direct lineage.

## What came before?

Follow the unsettled general-training claim.

## What changed?

Construct calibrated recovery in a structured count model.

## What did not work, or remains limited?

Matched-budget tests beat low-risk EMA but not CUSUM plus a restarted mean; model-specific rate optimality did not imply the best recovery performance.

## What comes next?

Test model mismatch and calibration error.

## Small attachment and reproduction scope

[ZIP]({{ "/_assets/optimizer-journey/34-count.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/34-count/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/34-count/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/34-count/core.py" | relative_url }})

[← 34]({{ "/research/en/34/" | relative_url }}) · [指引 / Index]({{ "/research/en/" | relative_url }}) · [34-task →]({{ "/research/en/34-task/" | relative_url }})
