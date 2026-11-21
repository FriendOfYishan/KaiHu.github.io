---
layout: research-post
title: "49 · Larger steps did not fix the direction"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: en
entry_id: 49
permalink: /research/en/49/
alternate_url: /research/zh/49/
---

Retrospective compiled on 2026-11-21. Original identifiers are preserved; adjacent numbers need not form a direct lineage.

## What came before?

Follow the viable low-state baseline.

## What changed?

Reduce signal suppression and add tensor RMS protection.

## What did not work, or remains limited?

Language NLL worsened relative to 48.

## What comes next?

Seek gains with an exact marginal-error identity.

## Small attachment and reproduction scope

[ZIP]({{ "/_assets/optimizer-journey/49.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/49/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/49/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/49/core.py" | relative_url }})

[← 48]({{ "/research/en/48/" | relative_url }}) · [指引 / Index]({{ "/research/en/" | relative_url }}) · [50 →]({{ "/research/en/50/" | relative_url }})
