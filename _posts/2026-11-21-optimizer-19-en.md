---
layout: research-post
title: "19 · Finite-horizon exponent scans"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: en
entry_id: 19
permalink: /research/en/19/
alternate_url: /research/zh/19/
---

Retrospective compiled on 2026-11-21. Original identifiers are preserved; adjacent numbers need not form a direct lineage.

## What came before?

Follow the asymptotic schedule relation.

## What changed?

Measure pre-update full-gradient tracking at finite horizons.

## What did not work, or remains limited?

Startup effects change the optimum and diagnostics cost gradients.

## What comes next?

Account for startup, geometry and temporal correlation.

## Small attachment and reproduction scope

[ZIP]({{ "/_assets/optimizer-journey/19.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/19/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/19/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/19/core.py" | relative_url }})

[← 18]({{ "/research/en/18/" | relative_url }}) · [指引 / Index]({{ "/research/en/" | relative_url }}) · [20 →]({{ "/research/en/20/" | relative_url }})
