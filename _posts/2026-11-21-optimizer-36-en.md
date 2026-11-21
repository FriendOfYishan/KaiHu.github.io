---
layout: research-post
title: "36 · Flash evidence momentum"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: en
entry_id: 36
permalink: /research/en/36/
alternate_url: /research/zh/36/
---

Retrospective compiled on 2026-11-21. Original identifiers are preserved; adjacent numbers need not form a direct lineage.

## What came before?

Follow the one-gradient constraint.

## What changed?

Release momentum when evidence justifies replacement.

## What did not work, or remains limited?

Cheaper kernels did not yield a clear end-to-end speedup.

## What comes next?

Constrain scale refresh by momentum dissipation.

## Small attachment and reproduction scope

[ZIP]({{ "/_assets/optimizer-journey/36.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/36/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/36/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/36/core.py" | relative_url }})

[← 35]({{ "/research/en/35/" | relative_url }}) · [指引 / Index]({{ "/research/en/" | relative_url }}) · [37 →]({{ "/research/en/37/" | relative_url }})
