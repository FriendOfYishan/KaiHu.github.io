---
layout: research-post
title: "33 · Ordinary harmonic averaging"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: en
entry_id: 33
permalink: /research/en/33/
alternate_url: /research/zh/33/
---

Retrospective compiled on 2026-11-21. Original identifiers are preserved; adjacent numbers need not form a direct lineage.

## What came before?

Follow the raw weighted-sum design.

## What changed?

Simplify harmonic averaging and scalar bookkeeping.

## What did not work, or remains limited?

Mismatched history weights introduce extra error and positivity protection.

## What comes next?

Use a reciprocal-residual identity.

## Small attachment and reproduction scope

[ZIP]({{ "/_assets/optimizer-journey/33.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/33/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/33/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/33/core.py" | relative_url }})

[← 32]({{ "/research/en/32/" | relative_url }}) · [指引 / Index]({{ "/research/en/" | relative_url }}) · [34 →]({{ "/research/en/34/" | relative_url }})
