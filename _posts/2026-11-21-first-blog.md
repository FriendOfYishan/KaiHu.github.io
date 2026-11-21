
---
layout: post
title: "我的第一篇技术博客"
date: 2026-11-21 14:00:00 +0800
categories: [MachineLearning, Research]
---

# 我的第一篇博客

今天开始在 GitHub 上记录技术研究。

## 1. 什么是随机梯度下降？

随机梯度下降（SGD）是一种常见的优化算法。

其更新公式为：

$$
\theta_{t+1} = \theta_t - \eta_t g_t
$$

其中：

- $\theta_t$ 表示模型参数
- $\eta_t$ 表示学习率
- $g_t$ 表示随机梯度

## 2. Python 示例

```python
import torch

x = torch.tensor([1.0], requires_grad=True)
loss = x.square().sum()
loss.backward()

print(x.grad)
```

## 3. 总结

后续将持续记录优化器实验和论文阅读。
