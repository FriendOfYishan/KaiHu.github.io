---
layout: research-post
title: "51-fused · Fusing candidate 51"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: en
entry_id: 51-fused
permalink: /research/en/51-fused/
alternate_url: /research/zh/51-fused/
---

Retrospective compiled on 2026-11-21. Original identifiers are preserved; adjacent numbers need not form a direct lineage.

## What came before?

Follow the quality study of 51.

## What changed?

Fuse the update and add a single-pass matrix path.

## What did not work, or remains limited?

The measured fused run still trailed fused AdamW in time; local FineWeb files are pilots.

## What comes next?

Complete the full FineWeb comparison before claiming scale transfer.

## Small attachment and reproduction scope

[ZIP]({{ "/_assets/optimizer-journey/51-fused.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/51-fused/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/51-fused/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/51-fused/core.py" | relative_url }})

[← 51]({{ "/research/en/51/" | relative_url }}) · [指引 / Index]({{ "/research/en/" | relative_url }})
