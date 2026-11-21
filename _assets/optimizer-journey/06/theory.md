# CrossAdamW v3：校准交叉残差、有效学习率变化约束与条件下降比较

日期：2026-11-08。状态：研究候选、推导及可运行参考实现；不代表已经达到 ICML 录用或大模型优于 AdamW 的证据要求。

本稿在 `01`–`05` 的基础上增加一个明确的新版本，不改写旧版本的定义与实验事实。实现见 [cross_adamw.py](../optimizers/cross_adamw.py)，新实验见 [20261108_calibrated](../experiments/20261108_calibrated/README_ZH.md)。本文的“二阶矩”是梯度平方的统计量，不是 Hessian 二阶优化。

**实验反馈：** 扩大同等学习率搜索后，v3 未胜过 AdamW，完整更新尾部幅度反而更大。数学结论应保留，固定收缩的性能主张应撤回；由此推导的下一步设计见 [07：下降预算门控](07_DESCENT_GATED_RESEARCH_ZH.md)，它的可靠在线界估计尚未完成。

## 1. 研究判断：应当修正什么

旧理论的条件风险、历史相关性分析和非平稳递推是有价值的。主要研究缺口有四个。

1. **较快指数不是相对优势。** `02` 中的 $\widetilde O(T^{-4/5})$ 依赖 $\sigma_t^2\le\upsilon\|\nabla F(x_t)\|^{3/2}$ 且无噪声底。这不是交叉项产生的性质；相同条件可以帮助其他算法。不能把它写成 CrossAdamW 超过 AdamW 的证明。
2. **残差系数不是实际噪声保留率。** 当 $h$ 来自带噪声的历史动量，$h^2+\kappa(g-h)^2$ 本身也含有历史噪声。已有推导覆盖了这一误差，但旧候选没有据此校准系数。
3. **固定区间只限制绝对大小。** $v\in[v_-,v_+]$ 没有给出相邻有效学习率变化的紧上限；旧理论的 $a_t\asymp\eta_t$ 也与常用固定 $\beta_1$ 实现不同。
4. **统计估计风险与下降量不同。** 必须直接比较相同状态、相同随机方向下的函数值，而不是仅比较两个方法各自的松上界。

因此本版把主张收缩为：明确校准二阶矩输入的目标；给出逐路径预条件器变化约束；在一般有界随机梯度下证明固定一阶动量的收敛上界；给出真实单步函数值比较的充分条件及失败反例。是否在网络上更快，交给公平训练实验。

## 2. 有限历史噪声校准

先研究**固定参数点**：$g_j=\mu+\xi_j$，创新独立同分布、均值零、方差 $s$。不要求对称或高斯。设 $p=\beta_1\in[0,1)$，历史长度 $n\ge1$，偏差修正的历史预测为

$$
h_n=\sum_{j=1}^n w_{n,j}g_j,\qquad
w_{n,j}=\frac{(1-p)p^{n-j}}{1-p^n}.
$$

它与下一次创新独立，$\mathbb Eh_n=\mu$，$\operatorname{Var}(h_n)=\rho_n s$，其中

$$
\boxed{\rho_n=\sum_j w_{n,j}^2
=\frac{1-p}{1+p}\frac{1+p^n}{1-p^n}.}
\tag{1}
$$

### 命题 1：唯一的非负残差校准

对 $q=h_n^2+c(g-h_n)^2$，有

$$
\mathbb Eq=\mu^2+[\rho_n+(1+\rho_n)c]s.
\tag{2}
$$

给定希望保留的噪声比例 $\kappa\ge\rho_n$，在这个输入族内，使 $\mathbb Eq=\mu^2+\kappa s$ 的唯一系数是

$$
c=\frac{\kappa-\rho_n}{1+\rho_n}\ge0.
\tag{3}
$$

证明：分别计算 $\mathbb Eh_n^2=\mu^2+\rho_ns$ 和 $\mathbb E(g-h_n)^2=(1+\rho_n)s$，代入即得。这是固定点的无条件期望等式，不是对训练历史条件化后的无偏性结论。

用一个无量纲参数 $\theta\in[0,1]$ 表示“高于历史动量噪声下限的保留比例”，令

$$
\boxed{\kappa_n=\rho_n+(1-\rho_n)\theta,\quad
c_n=\theta\frac{1-\rho_n}{1+\rho_n},\quad
q_t=h_t^2+c_{t-1}(g_t-h_t)^2.}
\tag{4}
$$

$\theta=1$ 对应原始二阶矩目标 $\mu^2+s$，**不代表逐步回到 AdamW**；$\theta=0$ 对应历史动量能量 $\mu^2+\rho_ns$。不能在保持 $c\ge0$ 的同一族内无偏估计 $\mu^2$，除非历史噪声为零。这是一个明确的正性与去噪程度之间的限制，并非所有估计器的不可行定理。

稳态 $\rho_\infty=(1-p)/(1+p)$，所以 $c_\infty=p\theta$。例如 $p=.9,\theta=.25$ 时，$\rho_\infty=1/19$，$c_\infty=.225$，实际目标是 $\mu^2+(11/38)s$，保留噪声约 $28.95\%$，不是 $25\%$。

原 $h^2+(g-h)^2$ 的稳态期望是 $\mu^2+(1+2\rho_\infty)s$；$p=.9$ 时噪声项被放大约 $10.53\%$。这解释了为什么“交叉项消噪”仍可能产生偏大的分母。

### 命题 2：校准后的正性与望远镜关系

对任意实数 $g,h$ 和 $c\ge0$，

$$
h^2+c(g-h)^2
=\frac{c}{1+c}g^2+(1+c)\left(h-\frac{c}{1+c}g\right)^2
\ge\frac{c}{1+c}g^2.
\tag{5}
$$

在稳态系数 $\theta=1,c=p$ 下，令 $\widetilde m=ph+(1-p)g$，还有

$$
\boxed{h^2+p(g-h)^2=g^2-\frac{\widetilde m^2-h^2}{1-p}.}
\tag{6}
$$

因此其校准可以解释为动量能量增量的控制变量。此恒等式使用固定系数递推；有限步偏差修正时，实际 $\widehat m_t$ 不等于这里的 $\widetilde m$，不能混用。

式 (5) 只控制输入 $q_t$。下面的延迟分母和变化限制会限制对当前突变的响应，不能据此宣称实际更新对任意梯度尖峰都有界。

### 命题 3：移动目标时保留完整误差

在真实训练的采样前历史 $\mathcal F_{t-1}$ 下，$h_t,\mu_t=\nabla_iF(x_t)$ 可测，令 $e_t=h_t-\mu_t$；当前噪声的二、三阶中心矩为 $s_t,k_t$，令 $W_t=\operatorname{Var}_{t-1}(\xi_t^2)$。对目标 $\nu_t^\kappa=\mu_t^2+\kappa_t s_t$，

$$
B_t=\mathbb E_{t-1}q_t-\nu_t^\kappa
=2\mu_te_t+(1+c_t)e_t^2+(c_t-\kappa_t)s_t,
$$
$$
\boxed{\mathbb E_{t-1}(q_t-\nu_t^\kappa)^2
=B_t^2+c_t^2(W_t-4e_tk_t+4e_t^2s_t).}
\tag{7}
$$

证明：展开 $q_t=\mu_t^2+2\mu_te_t+(1+c_t)e_t^2-2c_te_t\xi_t+c_t\xi_t^2$，对随机中心部分求方差即可。该式与旧理论 F2 属于同一输入族；本稿不把展开式冒称新发现。

训练中 $x_t$ 依赖过去创新，$\rho_ns_t$ 一般不再等于预测误差方差。因此 (1)–(4) 是有明确固定点依据的校准规则，不是神经网络训练中的无偏性保证。若在线选择 $\theta_t$ 且使其与历史预测误差相关，固定点无条件等式也不能直接继承。本版默认确定常数 $\theta$。

## 3. 完整可实现算法

给定 $p,b\in[0,1)$、$\theta\in[0,1]$、$0<v_-<v_+$、$\varepsilon\ge0$、初始化长度 $W\ge1$、固定参考学习率 $\eta_{\rm ref}>0$ 和变化参数 $\tau_0>0$。参数级状态只有未修正动量 $m$ 与**已修正且已投影的**二阶统计 $v$；每个参数张量另有步数。

初始化 $m_0=0,v_0=v_-\mathbf1$。第 $t$ 步先采样 $g_t$；在实际计算参数更新前，分母按以下顺序取值。

$$
\begin{aligned}
h_t&=m_{t-1}/(1-p^{t-1})\quad (t>W),\\
q_t&=\begin{cases}g_t^{\odot2},&t\le W,\\
h_t^{\odot2}+c_{t-1}(g_t-h_t)^{\odot2},&t>W,\end{cases}\\
\gamma_t&=\frac{1-b}{1-b^t},\quad
\tau_t=\tau_0\frac{\eta_t}{\eta_{\rm ref}},\\
\widetilde v_t&=\Pi_{[v_-,v_+]^d}[(1-\gamma_t)v_{t-1}+\gamma_tq_t],\\
v_t&=\begin{cases}\widetilde v_t,&t\le W,\\
\Pi_{[e^{-2\tau_t}v_{t-1},\,e^{2\tau_t}v_{t-1}]}(\widetilde v_t),&t>W,\end{cases}\\
D_t&=\begin{cases}\operatorname{diag}((\sqrt{v_t}+\varepsilon)^{-1}),&t\le W,\\
\operatorname{diag}((\sqrt{v_{t-1}}+\varepsilon)^{-1}),&t>W,\end{cases}\\
m_t&=p m_{t-1}+(1-p)g_t,\quad \widehat m_t=m_t/(1-p^t),\\
x_{t+1}&=(1-\eta_t\lambda)x_t-\eta_tD_t\widehat m_t.
\end{aligned}
\tag{8}
$$

逐坐标投影区间与固定区间有交集，因为旧 $v$ 已在固定区间内。前 $W$ 步为 AdamW 式初始化；仅在固定边界未激活时与标准 AdamW 完全一致。投影后不能再把 $v_t$ 当成未投影 EMA，或再除一次 $1-b^t$。

参考试验参数为 $p=.9,b=.999,\theta=.25,W=20,\tau_0=.02,\eta_{\rm ref}=.003,v_-=10^{-10},v_+=10^4,\varepsilon=10^{-8}$。这些是可复现实验起点，不是理论最优超参数或大模型推荐定值。大模型上尤其需要检查固定下界是否过早主导低幅值参数。

与 AdamW 同样每步一次新梯度，持久参数规模状态为 $2d$，逐坐标运算为 $O(d)$。这里没有额外 Hessian、旧点反传或跨设备统计归约。但本参考实现有额外临时张量和算子，未经融合，不声称同样墙钟时间或峰值显存。

## 4. 明确的步长稳定性保证

### 定理 1：逐路径相对变化约束

对 $t>W$，

$$
e^{-2\tau_t}\le v_{t,i}/v_{t-1,i}\le e^{2\tau_t},\qquad
\boxed{|\log D_{t+1,i}-\log D_{t,i}|\le\tau_t.}
\tag{9}
$$

第二个不等式在 $\varepsilon\ge0$ 下仍成立：对 $v\mapsto\log(\sqrt v+\varepsilon)$，关于 $\log v$ 的导数介于 0 与 $1/2$ 之间。因而实际坐标有效学习率 $\ell_{t,i}=\eta_t D_{t,i}$ 满足

$$
|\log\ell_{t+1,i}-\log\ell_{t,i}|
\le |\log(\eta_{t+1}/\eta_t)|+\tau_t.
\tag{10}
$$

恒定 $\eta=.003$ 时，每次预条件器变化倍率至多 $e^{.02}\approx1.0202$。这比单纯给出 $D$ 上下界更具体；它不是对 $\|x_{t+1}-x_t\|$ 或损失尖峰的同样限制。方向、外部学习率突变、权重衰减都需要另外考察。

该约束会牺牲突发变化时的响应速度，尤其分母上升受限可能延迟抑制新噪声。因此必须报告不加限制的消融、突发噪声实验与裁剪触发率，不能只展示平滑曲线。若改成仅限制分母下降，适应尖峰会更快，但下面关于双向变化的证明不再原样适用。

## 5. 固定一阶动量下的一般非凸上界

本节只证明 $\lambda=0$。假设 $F$ 是 $L$ 光滑且下有界，$\mathbb E_{t-1}g_t=\nabla F(x_t)$，$\|g_t\|\le G$ 几乎必然。相较只要求有界方差，这是更强的假设，不能声称优于 ADOPT 的假设范围。裁剪随机梯度通常会引入偏差，不能用裁剪默默满足无偏有界条件。

令

$$
d_-=(\sqrt{v_+}+\varepsilon)^{-1},\quad
d_+=(\sqrt{v_-}+\varepsilon)^{-1},\quad
R=(1-p)^{-1},\quad K=p/(1-p),\quad
A_t=\frac{\eta_tD_t}{1-p^t}.
$$

从 $t_0=W+1$ 开始，$A_t$ 在当前采样前可测。定义

$$
z_t=x_t-KA_{t-1}m_{t-1},\quad
\Delta A_t=A_t-A_{t-1},\quad
U=\sup_{t_0-1\le t\le T}\|A_t\|_{\rm op},
$$

以及使用确定性上界或对随机量取期望的

$$
S=\sum_{t=t_0}^T\mathbb E[\|A_t\|^2+K\|A_{t-1}\|\|A_t\|],\quad
V=\sum_{t=t_0}^T\mathbb E\|\Delta A_t\|,\quad
V_2=\sum_{t=t_0}^T\mathbb E\|\Delta A_t\|^2.
$$

### 定理 2：与实际固定 $\beta_1$ 对应的变差界

当 $U$ 有确定性上界时，

$$
\boxed{
d_-\sum_{t=t_0}^T\eta_t\mathbb E\|\nabla F(x_t)\|^2
\le \mathbb EF(z_{t_0})-F_*
+LG^2S+KG^2(1+LKU)V+LK^2G^2V_2.}
\tag{11}
$$

**证明。** 固定 $p$ 的递推给出精确恒等式

$$
z_{t+1}-z_t=-A_tg_t-K\Delta A_t m_{t-1},\qquad
\|z_t-x_t\|\le K\|A_{t-1}\|G.
\tag{12}
$$

用光滑下降式。因为 $A_t$ 可预测，主项为

$$
-\mathbb E_{t-1}\langle\nabla F(z_t),A_tg_t\rangle
\le-\eta_td_-\|\nabla F(x_t)\|^2
+LK\|A_{t-1}\|\|A_t\|G^2.
$$

同时 $\|\nabla F(z_t)\|\le G+LKUG$，所以变差的一次项不超过 $KG^2(1+LKU)\|\Delta A_t\|$。二次项由 $\|u+v\|^2\le2\|u\|^2+2\|v\|^2$ 至多为

$$LG^2\|A_t\|^2+LK^2G^2\|\Delta A_t\|^2.$$

求和、取全期望并用 $F(z_{T+1})\ge F_*$，得到 (11)。初始化阶段只通过 $z_{t_0}$ 和 $A_{t_0-1}$ 进入，未把其当前分母误当可预测。证毕。

**推论。** 令余下步数 $N=T-t_0+1$，使用恒定 $\eta\le\eta_{\max}$ 和 $\tau_t=C_\tau\eta$，其中 $C_\tau=\tau_0/\eta_{\rm ref}$ 与 $N$ 无关。由 (9)，

$$
\|D_t-D_{t-1}\|\le d_+(e^{C_\tau\eta}-1)
\le d_+C_\tau e^{C_\tau\eta_{\max}}\eta.
$$

因 $\eta/(1-p^t)$ 单调下降，

$$
U\le Rd_+\eta,\quad S\le(1+K)NU^2,
$$
$$
V\le Rd_+\eta+Rd_+C_\tau e^{C_\tau\eta_{\max}}N\eta^2,
\qquad V_2\le2UV.
\tag{13}
$$

代入 (11)，在初始化势能一致有界时得到

$$
\frac1N\sum_{t=t_0}^T\mathbb E\|\nabla F(x_t)\|^2
=O\!\left(\frac1{N\eta}+\eta+\frac1N\right).
\tag{14}
$$

因此 $\eta\asymp N^{-1/2}$ 给出 $O(N^{-1/2})$。不需要消失噪声，不需要改变 $p$，$b$ 可以固定。这是收敛上界，不是相对 AdamW 的更快指数，也不是实践中巨大 $d_+/d_-$ 下有用的定量常数。

**日程注意。** 证明要求 $C_\tau$ 在比较不同 $N$ 时固定，故实现中 `reference_lr` 是独立固定参数，不能随 $N$ 同步缩为实际学习率而仍宣称上述推论。未知时长用确定性非增 $\eta_t\asymp t^{-1/2}$，同样估计变差和平方步长求和可得带对数的界；常数步长训练只保证相应噪声邻域的上界。

**变化投影的统计代价。** `01` 的跟踪定理不能无条件用于这里随历史移动的投影区间。若统计目标不在该区间，投影未必缩短到目标的距离。因此本文没有宣称限制变化后仍保留未投影 EMA 的风险公式或无偏性。

## 6. 缩小分母何时真的使单步下降更快

以下比较两个方法在**同一个当前参数、同一个历史动量、同一次新梯度**下的下一步，而非已经分叉的两条训练轨迹。它给出函数值的直接比较，不是只比较各自上界。

令偏差修正动量可写为 $M=\bar m+a\xi$，$a=(1-p)/(1-p^t)$，$\bar m=\mathbb E_{t-1}M$，条件噪声协方差为 $\Sigma$。取两个采样前已确定的对角预条件器 $D_A,D_C$，令 $H=D_C-D_A$。它们可对应延迟 AdamW 和延迟 Cross 候选。标准 AdamW 的当前随机分母不满足此条件，不能在这里直接代入其总体均值冒充。

两步都使用相同权重衰减：

$$x_A=x-\eta(\lambda x+D_AM),\quad x_C=x-\eta(\lambda x+D_CM).$$

定义

$$
\mathcal A=\langle\mu,H\bar m\rangle,\quad
\mathcal B_A=\|\lambda x+D_A\bar m\|^2+a^2\operatorname{tr}(D_A^2\Sigma),
$$
$$\mathcal B_H=\|H\bar m\|^2+a^2\operatorname{tr}(H^2\Sigma).$$

### 定理 3：真实条件函数值比较

对 $L$ 光滑的 $F$，

$$
\boxed{\mathbb E_{t-1}[F(x_C)-F(x_A)]
\le-\eta\mathcal A+L\eta^2\left(\sqrt{\mathcal B_A\mathcal B_H}+\frac12\mathcal B_H\right).}
\tag{15}
$$

因而 $\mathcal A>L\eta(\sqrt{\mathcal B_A\mathcal B_H}+\mathcal B_H/2)$ 足以保证 Cross 的下一步**期望函数值更低**。

**证明。** 在 $x_A$ 处对 $x_C=x_A-\eta HM$ 使用光滑性。再用

$$\|\nabla F(x_A)-\mu\|\le L\eta\|\lambda x+D_AM\|,$$

得到逐样本上界 $-\eta\mu^THM+L\eta^2\|\lambda x+D_AM\|\|HM\|+(L\eta^2/2)\|HM\|^2$。取条件期望并用 Cauchy–Schwarz，得到 (15)。没有假设动量方向永远对齐真梯度。证毕。

**标量加速的透明特例。** 若 $\lambda=0,D_C=rD_A,r>1$，令 $A=\mu^TD_A\bar m$，$B=\|D_A\bar m\|^2+a^2\operatorname{tr}(D_A^2\Sigma)$，则

$$
\mathbb E_{t-1}[F(x_C)-F(x_A)]
\le-\eta(r-1)A+\frac{L\eta^2}{2}(r^2-1)B,
$$
$$\boxed{A>0,\quad 1<r<\frac{2A}{L\eta B}-1}\tag{16}$$

是充分收益区间。对 Hessian 为 $LI$ 的二次目标，上式为精确等式，边界也是精确的。

忽略 $\varepsilon$ 的理想稳态分母给出

$$r_i=\sqrt{\frac{\mu_i^2+s_i}{\mu_i^2+\kappa s_i}}\in[1,\kappa^{-1/2}].\tag{17}$$

但 (17) 不包含曲率、方向对齐与残余噪声，所以本身不能确保 (16)。当 $\mu=0$ 而动量有噪声时，二次目标上放大更新只会增加期望损失；这直接反驳了“缩小二阶矩永远更快更稳”。

定理 3 的 $L,\mu,\Sigma$ 在大模型上通常未知。本版**不声称每一步自动检验了这个证书**。它提供可检验的数学机制与测量目标；若以后根据探测梯度选择参数，要计入其计算成本、置信误差和选择偏差。

## 7. 权重衰减、泛化与大模型边界

算法包含解耦衰减，定理 3 也显式保留它；定理 2 的全局驻点结论仅覆盖 $\lambda=0$。$\lambda>0$ 的平衡通常满足 $D\nabla F(x)+\lambda x=0$，不等同于 $\nabla F(x)=0$。不能把带衰减实验写成定理 2 的直接验证。

更快达到训练损失不等于更好的验证损失；预条件器平滑不等于参数更新平滑，更不等于没有 loss spike。需要同时报告训练/验证曲线、达标 tokens/时间、更新范数分位数、有效学习率对数变化、失败次数和显存。

本地实际只执行 CPU digits MLP，不能据此声称改善 Transformer 或大模型预训练。代码支持常规 PyTorch 参数及低精度参数的 FP32 状态，但 AMP、分布式、参数分片、融合内核和大模型负载尚未验证。

## 8. 新颖性边界与原始文献

以下为截至 2026-11-08 的有针对性的原始文献核查，不是穷尽检索或优先权证明。

|已有工作|与本设计的交集|本稿必须区分的部分|
|---|---|---|
|[AdaBelief](https://arxiv.org/abs/2010.07468)|残差平方参与预条件|其残差围绕更新后动量；本稿保留信号能量并校准历史噪声，不能只靠“用残差”主张创新|
|[AdaMomentum](https://arxiv.org/abs/2106.11514)|对动量的平方做第二次 EMA|本稿 $\theta=0$ 接近这个思想，但时序、偏差修正、投影不同；不能把动量能量归一化称为首次提出|
|[ADOPT](https://arxiv.org/html/2411.02853v3)|延迟二阶统计，消除当前分母相关|ADOPT 还改变归一化与动量顺序；本稿保留原始梯度动量并限制分母变化，延迟本身不是新贡献|
|[AdamS](https://arxiv.org/html/2505.16363v1)|动量与梯度能量组合及 LLM 实验|它不存储 Adam 式第二状态；本稿依然为两个状态，不能声称更省内存|
|[CAME](https://arxiv.org/abs/2307.02047)|置信信息用于稳定大模型优化|本稿不是因子化的置信预条件，也没有同等规模证据|
|[MVN-Grad](https://arxiv.org/abs/2602.10204)|方差归一化及动量顺序|需要比较不同顺序和目标；不能以“降低分母方差”笼统区别|
|[EXAdam](https://arxiv.org/abs/2412.20302)|cross-moments 命名及偏差修正|需要公式级区别，名称不是创新|
|[Adam EoS 分析](https://arxiv.org/abs/2608.20638)|自适应二阶矩与稳定性反馈|限制分母变化未必改善整个反馈系统，应测量曲率与梯度反转|

已有一般随机非凸 oracle 下界也不允许仅更换二阶矩就在同一问题类上宣称无条件更快复杂度指数：[Arjevani et al.](https://arxiv.org/abs/1912.02365)。

可争取的论文贡献应是“有限历史校准如何改善实际分母与下降权衡，以及何时失败”，并配套机制测量和跨规模实证。式 (4) 仍属于旧文的代数输入族，投影、动量变换及光滑性比较也属于标准工具。若实验优势在学习率重调后消失，或 AdaMomentum/AdamS/ADOPT 同样解释结果，仅这组公式不足以构成强 ICML 方法论文。

## 9. 投稿前的判定实验

第一层为本次完成的小型机制试验：固定初始化/采样，等学习率、等梯度预算、等学习率搜索预算；校准消融、去掉变化限制、当前分母、原始平方输入、同目标混合输入和纯历史动量能量。

第二层应使用至少一种视觉网络和一个标准 Transformer 语言建模任务。对 AdamW、AdaBelief、AdaMomentum、ADOPT、AdamS，以及适合该模型的 Muon/MARS，使用同模型、同数据顺序、同 tokens、同精度与权重衰减分组，等额调参。报告配对种子差值与独立运行误差，不能把同模型多个 batch 当作独立试验。

第三层才是大模型主张：建议先做 100M–300M 参数语言模型的预训练，再根据结果决定是否扩大到约 1B；这是实验分级建议，不是已启动任务或预估费用。预先固定 token 预算、验证集、达标损失和故障定义。统计所有额外探测计算，验证状态字节数及端到端吞吐，不用优化器微基准替代训练时间。

最有区分度的否证项是：AdamW 充分调整学习率后优势是否还在；同目标混合/动量平方是否一样好；噪声突然上升或方向快速变化时上升限幅是否造成更大 spike；收益是否仅来自放大整体步长。不能通过事后筛选任务、种子或阈值来回避这些问题。

## 10. 当前能写与不能写的结论

能写：给出两状态、一次梯度调用的明确候选；固定点有限历史校准；相邻有效学习率的逐路径约束；固定动量的一般非凸上界；同状态下真实期望损失比较的充分条件及二次反例；真实 CPU pilot 结果。

不能写：普遍严格快于 AdamW；改变一般随机非凸最优复杂度；在所有噪声/漂移下更稳；已在大模型训练上获胜；已完成穷尽新颖性审查；保证 ICML 录用。
