---
layout: research-post
title: "37 · Dissipation-paid scale refresh"
date: 2026-11-21 12:00:00 +0800
categories: [optimizer-research]
lang: en
entry_id: 37
permalink: /research/en/37/
alternate_url: /research/zh/37/
---

Retrospective compiled on 2026-11-21. Original identifiers are preserved; adjacent numbers need not form a direct lineage.

## What came before?

Follow the controller overhead.

## What changed?

Pay for scale changes with momentum dissipation.

## What did not work, or remains limited?

Theory and synthetic checks do not establish Transformer quality.

## What comes next?

Compress scales per layer and measure the information loss.

## Small attachment and reproduction scope

[ZIP]({{ "/_assets/optimizer-journey/37.zip" | relative_url }}) · [README]({{ "/_assets/optimizer-journey/37/README.md" | relative_url }}) · [报告 / Report]({{ "/_assets/optimizer-journey/37/report.md" | relative_url }}) · [代码 / Code]({{ "/_assets/optimizer-journey/37/core.py" | relative_url }})

[← 36]({{ "/research/en/36/" | relative_url }}) · [指引 / Index]({{ "/research/en/" | relative_url }}) · [37-batch →]({{ "/research/en/37-batch/" | relative_url }})
