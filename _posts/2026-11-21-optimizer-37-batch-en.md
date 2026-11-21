---
layout: research-post
title: "37-batch · Layer37-B and gradient accounting"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: en
entry_id: 37-batch
permalink: /research/en/37-batch/
alternate_url: /research/zh/37-batch/
---

Retrospective compiled on 2026-11-21. Original identifiers are preserved; adjacent numbers need not form a direct lineage.

## What came before?

Follow Layer37 compression.

## What changed?

Increase effective batch size gradually.

## What did not work, or remains limited?

A faster iteration rate costs gradients and does not restore geometry.

## What comes next?

Compare equal-token and equal-time quality.

## Small attachment and reproduction scope

[ZIP]({{ "/_assets/optimizer-journey/37-batch.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/37-batch/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/37-batch/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/37-batch/core.py" | relative_url }})

[← 37]({{ "/research/en/37/" | relative_url }}) · [指引 / Index]({{ "/research/en/" | relative_url }}) · [37-layer →]({{ "/research/en/37-layer/" | relative_url }})
