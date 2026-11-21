---
layout: research-post
title: "15 · Making preview mixing well defined"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: en
entry_id: 15
permalink: /research/en/15/
alternate_url: /research/zh/15/
---

Retrospective compiled on 2026-11-21. Original identifiers are preserved; adjacent numbers need not form a direct lineage.

## What came before?

Follow the AWA definition audit.

## What changed?

Repair invalid preview-mixture weights.

## What did not work, or remains limited?

Original weights can make moments negative or divide by zero.

## What comes next?

Study filter bias with explicit computation costs.

## Small attachment and reproduction scope

[ZIP]({{ "/_assets/optimizer-journey/15.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/15/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/15/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/15/core.py" | relative_url }})

[← 14]({{ "/research/en/14/" | relative_url }}) · [指引 / Index]({{ "/research/en/" | relative_url }}) · [16 →]({{ "/research/en/16/" | relative_url }})
