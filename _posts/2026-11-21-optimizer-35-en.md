---
layout: research-post
title: "35 · Separating the two memories"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: en
entry_id: 35
permalink: /research/en/35/
alternate_url: /research/zh/35/
---

Retrospective compiled on 2026-11-21. Original identifiers are preserved; adjacent numbers need not form a direct lineage.

## What came before?

Follow the moment ablations.

## What changed?

Retain EMA momentum and test longer scale memory.

## What did not work, or remains limited?

Small gains changed sign with learning rate; VR costs extra oracle calls.

## What comes next?

Explore evidence-triggered momentum with one gradient.

## Small attachment and reproduction scope

[ZIP]({{ "/_assets/optimizer-journey/35.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/35/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/35/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/35/core.py" | relative_url }})

[← 34-task]({{ "/research/en/34-task/" | relative_url }}) · [指引 / Index]({{ "/research/en/" | relative_url }}) · [36 →]({{ "/research/en/36/" | relative_url }})
