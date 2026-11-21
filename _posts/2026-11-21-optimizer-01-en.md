---
layout: research-post
title: "01 · Formalizing v2.1"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: en
entry_id: 01
permalink: /research/en/01/
alternate_url: /research/zh/01/
---

Retrospective compiled on 2026-11-21. Original identifiers are preserved; adjacent numbers need not form a direct lineage.

## What came before?

Start from the historical cross term.

## What changed?

Account for EMA history and drift.

## What did not work, or remains limited?

Fixed-point denoising does not establish training gains.

## What comes next?

Define a predictable preconditioner and memory schedule.

## Small attachment and reproduction scope

[ZIP]({{ "/_assets/optimizer-journey/01.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/01/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/01/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/01/core.py" | relative_url }})

[指引 / Index]({{ "/research/en/" | relative_url }}) · [02 →]({{ "/research/en/02/" | relative_url }})
