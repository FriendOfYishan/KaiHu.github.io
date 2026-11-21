# 候选 51 的等价简式与 Triton 实现

`C51Slim` 是可读的 PyTorch 参考实现。`C51Fused` 提供相同的矩阵、向量、首步和解耦权重衰减公式。`alpha=.5, tau=2.` 对应原版 51。`alpha=0., tau=2.` 对应原消融 `51nf`。两者都接受原版 51 的参数分组与 full 模式检查点。加载 nofloor 检查点时显式传入 `alpha=0.`，原实现没有把 mode 存入优化器状态。

## 训练入口

```python
from optzoo import split_groups
from c51_fused import C51Fused

params = list(model.parameters())
optimizer = C51Fused(split_groups(params, wd=0.01), lr=3e-4, alpha=.5, tau=2.)
```

Triton 版本使用连续的 CUDA FP32 参数与梯度。实验中的前向和反向沿用 BF16 混合精度。二维及更高维参数按第一维分行，其余维展平为列。矩阵最多支持每轴 65,536 个元素和有符号 32 位矩阵索引。向量和标量使用逐坐标残差。更换实现时沿用相同的学习率日程、梯度缩放、梯度范数裁剪与衰减分组。

## 增益与截断

设历史校正动量为 $B$，校正后的残差行列均值为 $R_i,C_j$。令 $a_i=\operatorname{mean}_j B_{ij}^2+R_i$，$b_j=\operatorname{mean}_i B_{ij}^2+C_j$，$v_0=\operatorname{mean}_i a_i$，$e=10^{-16}$。同一个函数作用于两个轴

$$
\phi(v)=\frac12\left(\frac{\alpha}{v_0+e}+\frac{1-\alpha}{v+e}\right),
\qquad K_{ij}^2=\phi(a_i)+\phi(b_j).
$$

取 $\alpha=1/2$ 后，全张量、行、列三项的系数恰为 $1/2,1/4,1/4$。实现将两份全张量常数合并。逐坐标更新使用

$$
U_{ij}=\frac{B_{ij}}{\sqrt{B_{ij}^2+R_i C_j/\max(\operatorname{mean}R,10^{-30})+e}}
+\operatorname{clip}\!\left(\frac{1-\beta}{1-\beta^t}(G_{ij}-B_{ij})K_{ij},-\tau,\tau\right).
$$

状态更新的残差取 $G-M_{t-1}$，创新项取 $G-B$。首步使用 $G/\sqrt{\max(G^2,\operatorname{mean}G^2)+e}$。分组的最终更新为 $W\gets(1-\eta\lambda)W-\eta U$。

令 $N$ 为同一残差平方 EMA 的稠密矩阵，$\widetilde N$ 为行列重建。在重建分母保护未触发的精确算术下，$D=N-\widetilde N$ 的各行、各列和均为零，因此

$$
\sum_{ij} K_{ij}^2 D_{ij}
=\sum_i\phi(a_i)\sum_j D_{ij}+\sum_j\phi(b_j)\sum_i D_{ij}=0.
$$

这条恒等式对每步任取的张量级 $\alpha\in[0,1]$ 成立，约束的是截断前的平方增益预算。逐坐标历史项和截断后的实际下降量还取决于梯度方向。

## 验证与复现

在原 51 的 GPU 环境中，把本目录的 Python 文件放进同一目录，从该目录运行以下命令。参数是旧实验里同时包含 `c51.py`、`optzoo.py` 和 `tasks.py` 的目录。脚本沿用旧实验的真实 WikiText-2 数据与模型，每种方法计时前先用同形状训练步预热。

```bash
python verify51.py /root/autodl-tmp/candidate51
python verify51.py /root/autodl-tmp/candidate51 --methods 51fused --seed 22 --steps 4883 --lr .0003 --output long_default
python verify51.py /root/autodl-tmp/candidate51 --methods 51fused --seed 22 --steps 4883 --lr .0003 --end-alpha .75 --end-tau 1.5 --output long_taper
```

另两个短预算种子用 `--methods 51fused --seed 11 --output seed11` 和 `--methods 51fused --seed 33 --output seed33`。长预算变体从 40% 进度起线性插值到给定终值。每步只改参数组的两个标量。固定参数和可调参数共享内核。

矩阵每步依次执行块内部分归约、两轴汇总、全张量标量归约、逐坐标写回。最后一次写回融合历史归一化、创新增益、截断、权重衰减和动量写回。向量每步一次发射。临时工作区在同一 CUDA 流上复用，计入峰值显存；检查点只存原 51 的状态。

## RTX 3090 实测

使用原实验的 30,555,136 参数 WikiText-2 Transformer、PyTorch 2.8.0+cu128 和 Triton 3.4.0。短预算是种子 22、503 步、4,120,576 tokens、峰值学习率 0.001、权重衰减 0。训练计时继承原脚本；完整墙钟另计初始化和全部评估。各方法先预热，完整墙钟均排除首次 JIT。

|实现|训练计时 s|完整墙钟 s|优化器计时 s|状态 B/参数|峰值 GiB|验证 NLL|
|---|---:|---:|---:|---:|---:|---:|
|原 51|36.53|37.35|13.94|4.012|3.207|5.47279|
|融合 51|26.40|27.22|3.66|4.012|3.212|5.47798|
|AdamW fused|24.08|24.89|1.10|8.000|3.323|5.52744|

同一预生成梯度序列下，21 步逐步比较原 51 与融合实现，默认参数的最大权重绝对误差为 $1.67\times10^{-6}$。训练开启 BF16 混合精度后，两个实现的舍入差异会进入后续梯度。

相同 503 步日程下，融合版种子 11、22、33 的验证 NLL 分别为 **5.44792、5.47798、5.47673**，均值 **5.46754**。工作区同配置原 51 均值为 5.4718，AdamW 为 5.5313，融合版逐种子保留了相对 AdamW 的优势。补跑两个种子的逐检查点结果在 `seed11.json` 和 `seed33.json`。

长预算使用同一数据、种子 22、4,883 步、峰值学习率 0.0003、权重衰减 0。后段变体前 40% 沿用默认值，此后线性提高全张量份额并收紧截断。两条曲线在前四个检查点逐值相同。

|融合 51 设置|最佳验证 NLL（约 20M）|40M 终点验证 NLL|训练计时 s|
|---|---:|---:|---:|
|固定 0.5 / 2|4.688754|5.058028|247.89|
|后段调至 0.75 / 1.5|4.688815|5.058542|248.64|

本次配对继续支持固定参数为默认。原始检查点见本目录的 `results.json`、`long_default.json` 和 `long_taper.json`。

## FineWeb-Edu 与单遍读写

`C51Fused(..., single_pass=True)` 在写入 $M_t$ 和权重时一并累加 $M_t^2$、$(G_t-M_{t-1})^2$ 的块内行列和。接下来的核更新残差 EMA、把 $M_t^2$ 的边际除以 $(1-\beta^t)^2$，再归约下一步用的全张量标量。稳态每步一次矩阵读写，仍有两次小统计量核发射。首步需要独立计算当前梯度均方。缓存可以从优化器检查点重建，额外占用 $O(m+n)$ 显存。

迁移自原版融合状态并继续更新的数值检查覆盖 $16384\times768$ embedding、两个 FFN 方向、尾块以及向量参数。RTX 3090 上 18 步的最大权重绝对差为 $4.77\times10^{-7}$。

长跑用 13 层、宽度 768 的 105,030,144 参数 Transformer，BF16 autocast 与 FP32 参数。加载项目现有 FineWeb-Edu 文档级划分和 BPE，每个训练运行从头连续读取 299,999,232 个预测目标，约为训练集的 42.6%。种子 22 与 33 各自配对同一初始化和数据；两种方法各用种子 7 搜索同一学习率网格，主实验沿同一套 300M token 日程记录 40M、100M 等检查点。每个检查点报告区间训练 NLL、固定训练探针 NLL、验证 NLL、累计训练与优化器耗时和峰值显存。

| 105M 模型，预热后 52 步 | 整步墙钟 s | 优化器累计 s | 峰值 allocated MiB |
|---|---:|---:|---:|
| 51 原融合核 | 6.987 | 0.685 | 6,097 |
| 51 单遍融合核 | 6.873 | 0.540 | 6,088 |
| AdamW fused | 6.681 | 0.386 | 6,486 |

51 的持久张量与派生缓存合计 422,425,008 字节（4.022 B/参数），AdamW 的状态为 840,241,584 字节（8.000 B/参数）。峰值显存包含工作区和激活。实测数据在 `fineweb_results/`。

```bash
python fineweb51.py /root/autodl-tmp/candidate45_long
```
