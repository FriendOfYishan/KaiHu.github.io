# 2. CrossAdamW：低状态开销、对称噪声与自适应记忆的条件理论

这是基于 v2.1 思想构造的新候选，**不是原 v2.1 更新已经拥有的新定理**。本章提供完整数学算法与证明，不提供训练实现。核心结论是有额外结构条件的平均梯度平方速率。

## 2.1 设计中哪些地方改变了

原 v2.1 用当前二阶矩作分母、固定动量、偏差修正。这里实际使用上一拍的有界分母；方向动量的新信息权重与步长同阶；一阶状态从零初始化并直接递推，不用固定 beta 的偏差修正。原来的交叉项保留在二阶矩输入中。额外收缩是显式可选项。

分清三个角色：

- $a_t=1-\beta_{1,t}$：方向和历史预测器的新信息权重。
- $\gamma_t=1-\beta_{2,t}$：二阶矩的遗忘率。
- $\kappa_t$：保留多少残差能量；$\alpha_t$：消除多少信号—噪声交叉项。

这些量不能混成一个“去噪系数”。高 beta 的固定点统计收益也不能代替方向的动态跟踪证明。

## 2.2 完整数学算法

给定 $0<\underline v<\overline v<\infty$、$\varepsilon\ge0$，定义
$$
d_-=(\sqrt{\overline v}+\varepsilon)^{-1},\qquad
d_+=(\sqrt{\underline v}+\varepsilon)^{-1}.
\tag{A1}
$$

初始化 $m_0=0$、$v_0\in[\underline v,\overline v]^d$、标量 $z_0\in[0,1]$。在采样前确定
$$
\eta_t>0,\quad a_t\in[c_-\eta_t,c_+\eta_t]\subset(0,1/2],
\quad \gamma_t\in[\gamma_-,\gamma_+]\subset(0,1],
$$
$$
\alpha_t\in[0,1],\quad \kappa_t\in[\kappa_-,1],\quad 0<\kappa_-\le1.
\tag{A2}
$$
$a_t$ 为全局标量，允许依赖历史。$\alpha_t$,$\kappa_t$可为标量或逐坐标可预测量；以下统计公式逐坐标适用。

每步只取一次固定大小小批量梯度：
$$
\begin{aligned}
h_t&=m_{t-1},\quad
D_t=\operatorname{diag}((\sqrt{v_{t-1}}+\varepsilon)^{-1}),\\
g_t&=g(x_t,\xi_t),\\
q_t&=\kappa_t[g_t^{\odot2}-2\alpha_t h_t\odot(g_t-h_t)]
+(1-\kappa_t)h_t^{\odot2},\\
m_t&=(1-a_t)m_{t-1}+a_t g_t,\\
x_{t+1}&=x_t-\eta_t D_t m_t,\\
v_t&=\Pi_{[\underline v,\overline v]^d}
[(1-\gamma_t)v_{t-1}+\gamma_t q_t].
\end{aligned}
\tag{A3}
$$
这里 $D_t$必须使用旧 v；不能在实现时无声换成当前 v。$\kappa=1$ 精确还原原 v2.1 的二阶矩输入，$\alpha=0$ 还原 $\lambda=1-\kappa$ 的 V2 输入。对称噪声候选取 $\alpha_t=1$，于是
$$
\boxed{q_t=h_t^{\odot2}+\kappa_t(g_t-h_t)^{\odot2}.}
\tag{A4}
$$
它不需要在线估计三阶矩或增加参数规模状态。

**一个完全指定的低状态门控。** 固定 $\omega\in(0,1],\tau>0$，采样后计算
$$
z_t=(1-\omega)z_{t-1}+\omega\min\{1,\|g_t-h_t\|^2/(\|h_t\|^2+\tau)\}.
$$
下一步使用
\[
a_t=[c_-+(c_+-c_-)z_{t-1}]\eta_t,\quad
\gamma_t=\gamma_-+(\gamma_+-\gamma_-)z_{t-1},
\]
\[
\alpha_t=1,\qquad
\kappa_t=\kappa_-+(1-\kappa_-)z_{t-1}.
\tag{A5}
\]
大残差让下一步更快忘记旧统计，并减少收缩。这个门控是可实施的启发式，不声称识别了真实漂移或达到风险 oracle。残差也包含随机噪声。所有收敛结论对任意满足 (A2) 的可预测门控成立，不依赖 (A5) 的统计最优性。取 \(\kappa_-=1\) 就不改变二阶矩的原目标。

## 2.3 定理 F：增强对称噪声处理的精确收益与边界

对一个坐标，设 \(g=\mu+\xi,h=\mu+e\)，当前条件噪声满足均值零，方差 s、三阶矩 k、\(W=\operatorname{Var}(\xi^2)\)。对 (A4)，
\[
q_\kappa=\mu^2+2\mu e+(1+\kappa)e^2-2\kappa e\xi+\kappa\xi^2.
\tag{F1}
\]
因此对原目标 \(\nu=\mu^2+s\)，
\[
B_\kappa=\mathbb E q_\kappa-\nu
=2\mu e+(1+\kappa)e^2-(1-\kappa)s,
\]
\[
\boxed{R_\kappa^\nu=B_\kappa^2+
\kappa^2(W-4ek+4e^2s).}
\tag{F2}
\]

**证明。** 对 (F1) 取均值；随机中心部分为
\(\kappa(\xi^2-s-2e\xi)\)，其方差直接展开即得。证毕。

当 h 准确、k=0 时，存在两种必须分开的结论：

**同一个噪声折扣目标。** 设 \(\theta_\kappa=\mu^2+\kappa s\)。V2 输入 \(\kappa g^2+(1-\kappa)h^2\) 的风险为 \(\kappa^2(4\mu^2s+W)\)，而 (A4) 为 \(\kappa^2W\)。因此风险严格减少 \(4\kappa^2\mu^2s\)，除非 \(\kappa\mu s=0\)。这保留了 v2.1 的交叉噪声消除思想，并允许进一步压低偶次残差的方差。

**与原 v2.1 比较同一个未折扣目标。** 必须计入新增偏差：
\[
R_\kappa^\nu=\kappa^2W+(1-\kappa)^2s^2,
\quad R_1^\nu=W.
\tag{F3}
\]
若 \(W+s^2>0\)，单步 oracle 是
\[
\kappa_*={s^2\over s^2+W},\qquad
R_*^\nu={Ws^2\over W+s^2}.
\]
对 \(0\le\kappa<1\)，严格改善当且仅当
\[
\boxed{(1+\kappa)W>(1-\kappa)s^2.}
\tag{F4}
\]
这是把 (F3) 与 W 相减后分解 \(1-\kappa\) 得到，不是无条件“更强降噪”。例如高斯噪声、h 准确时，\(\kappa=1/2\) 的同目标单步风险为 \(3s^2/4\)，小于原来的 \(2s^2\)；但 Rademacher 噪声 \(W=0\) 时原输入风险为零，任何 \(\kappa<1,s>0\) 都更差。

**时间平均必须重新调节收缩。** 理想预测器、固定点 iid 样本、确定权重 \(w_j\ge0,\sum w_j=1\) 下，令 \(\rho=\sum w_j^2\)，则
\[
R_{\kappa,\rho}^\nu=\rho\kappa^2W+(1-\kappa)^2s^2,
\quad \kappa_{*,\rho}=s^2/(s^2+\rho W).
\tag{F5}
\]
因此长窗口下最优 \(\kappa\) 更接近 1；不能将高斯单步 \(\kappa_*=1/3\) 不加调整用于长 EMA。实际历史预测器下则应用第 1 章 (9) 的嵌套参数，或 (10)–(11) 的非平稳递推；不能使用理想独立式冒充实际风险。

**可选的极端残差限制。** 把 (A4) 的残差平方换成 \(\min\{(g-h)^2,R^2\}\)，仍是非负偶函数、O(d) 运算。h 准确时，令 \(Y=\xi^2\)、\(\phi(Y)=\min(Y,R^2)\)，独立复制 \(Y'\) 给出
\[
\operatorname{Var}\phi(Y)=\tfrac12\mathbb E(\phi(Y)-\phi(Y'))^2
\le\tfrac12\mathbb E(Y-Y')^2=\operatorname{Var}Y.
\]
但原目标 MSE 是 \(\kappa^2\operatorname{Var}\phi(Y)+(\kappa\mathbb E\phi(Y)-s)^2\)，有额外截断偏差。本选项不属于默认 (A3)，也不是免费增强。主收敛定理只使用 D 的可预测有界性，因此允许此选项，但不提供它的训练优势。

## 2.4 长历史的分母稳定性

由投影逐路径有
\[
d_-I\preceq D_t\preceq d_+I.
\tag{S1}
\]
不会因为二阶矩偶然接近零而出现无限有效步长。对 \(u,v\in[\underline v,\overline v]\)，
\[
|(\sqrt u+\varepsilon)^{-1}-(\sqrt v+\varepsilon)^{-1}|
\le C_D|u-v|,
\quad C_D={1\over2\sqrt{\underline v}(\sqrt{\underline v}+\varepsilon)^2}.
\tag{S2}
\]
证明是求导并在区间上取最大绝对值。这次严格正下界是算法的一部分，未在证明里额外偷加。

若目标二阶矩在该区间内，第 1 章 (11)–(12) 控制含漂移、预测偏差及噪声的跟踪误差。将 (S2) 平方取期望可把它转成分母误差上界。\(\gamma_->0\) 防止二阶统计的时间记忆无限延长；在给定外部输入、相同遗忘日程下，旧 v 初值的直接影响最多按 \((1-\gamma_-)^t\) 衰减。若门控会随状态变化，不能把这个输入固定的收缩式误称整个闭环系统的收缩定理。

当输入均值发生确定跳变且此后保持常数、无预测偏差、固定 \(\gamma\) 时，均值误差恰按 \((1-\gamma)^k\) 衰减，恢复至比例 \(\delta\) 需要
\(k\ge\log(1/\delta)/[-\log(1-\gamma)]\)。这刻画有限记忆的恢复速度，不声称未知漂移被门控完美识别。

代价是预条件器不再追求任意大的动态范围，可能损失某些病态问题上的自适应优势；常数 \(d_+/d_-\) 也会进入收敛界。

## 2.5 收敛假设、常数和目标

**H1.** \(F:\mathbb R^d\to\mathbb R\) 为 L 光滑，L>0，且 \(F\ge F_*>-\infty\)。不要求凸性、PL 条件或已知最优解。

**H2.** 条件无偏，\(\mu_t=\nabla F(x_t)\)，
\[
\mathbb E_{t-1}g_t=\mu_t,\quad
\sigma_t^2=\mathbb E_{t-1}\|g_t-\mu_t\|^2<\infty.
\]
主速率另要求
\[
\boxed{\sigma_t^2\le\sigma_0^2+\upsilon\|\mu_t\|^{3/2}.}
\tag{H3}
\]
精确目标速率取 \(\sigma_0=0\)。非零 \(\sigma_0\) 的影响显式保留。此条件不由分布对称、有限四阶矩或二阶矩 MSE 改善推出。

**H4.** 参数顺序满足 (A1)–(A3)，\(\eta_t\) 为确定性正步长。为简写定义
\[
C={d_+^2\over d_-}+{d_-\over4},\quad
\Lambda={2C\over c_-},\quad B={Ld_+^2\over2}.
\tag{C1}
\]
要求
\[
c_-^2\ge {40CL^2d_+^2\over d_-},\quad c_+\ge c_->0,
\quad \eta_t\le\eta_0\le\min\{1/(2c_+),d_-/(4Ld_+^2)\}.
\tag{C2}
\]
这些是足够条件，可能保守；没有暗示实践中最佳学习率如此小。它们是可满足的固定常数，不依赖总步数 T。

例如取 \(L=1,\varepsilon=0,\underline v=1/4,\overline v=1\)，则 \(d_-=1,d_+=2\)。取 \(c_-=32,c_+=64,\eta_0=1/128\) 满足全部条件，得到 \(C=17/4,\Lambda=17/64,K=4353/2\)。这是条件相容性的代数例子，不是推荐训练参数。

令采样前的方向预测误差
\[
r_t=m_{t-1}-\mu_t,\quad
\mathcal L_t=F(x_t)-F_*+\Lambda\|r_t\|^2,
\quad K=c_+^2(B\eta_0^2+2\Lambda).
\tag{C3}
\]
本章定理首先针对 \(w=0\)。\(\mathcal L_1=F(x_1)-F_*+\Lambda\|\nabla F(x_1)\|^2\) 有限。有限二阶矩与光滑性足以保证每个有限步的相关期望有限；亦可先作停止时间局部化再取极限。统计章节的四阶矩要求不属于本收敛证明的必要条件。

## 2.6 引理 G：一次下降中的真实动量误差

记条件期望为 \(\mathbb E_{t-1}\)。在 H1、H2、H4 下，
\[
\mathbb E_{t-1}[F(x_{t+1})-F(x_t)]
\le-{d_-\over2}\eta_t\|\mu_t\|^2+
C\eta_t\|r_t\|^2+B\eta_t^2a_t^2\sigma_t^2.
\tag{G1}
\]

**证明。** 当前动量写为
\(m_t=\mu_t+(1-a_t)r_t+a_t\xi_t\)。因为 D、a、r 在采样前确定，
\[
\mathbb E_{t-1}\|m_t\|^2
=\|\mu_t+(1-a_t)r_t\|^2+a_t^2\sigma_t^2
\le2\|\mu_t\|^2+2\|r_t\|^2+a_t^2\sigma_t^2.
\]
由光滑性，目标变化不超过
\(-\eta_t\langle\mu_t,D_t\mathbb E_{t-1}m_t\rangle+
Ld_+^2\eta_t^2\mathbb E_{t-1}\|m_t\|^2/2\)。交叉项满足
\[
|\langle\mu_t,D_t(1-a_t)r_t\rangle|
\le {d_-\over4}\|\mu_t\|^2+{d_+^2\over d_-}\|r_t\|^2.
\]
使用 \(Ld_+^2\eta_t\le d_-/4\) 后，梯度平方系数至多 \(-d_-\eta_t/2\)，误差系数至多 C\(\eta_t\)，得到 (G1)。证毕。

## 2.7 引理 H：历史记忆与参数漂移的联合跟踪

在同样条件下，
\[
\boxed{\mathbb E_{t-1}\|r_{t+1}\|^2
\le(1-a_t)\|r_t\|^2+
{5L^2d_+^2\eta_t^2\over a_t}\|\mu_t\|^2
+2a_t^2\sigma_t^2.}
\tag{H1}
\]

**证明。** 令 \(e_t=m_t-\mu_t\)，则
\(r_{t+1}=e_t+(\mu_t-\mu_{t+1})\)。\(\mu_{t+1}\) 依赖当前随机梯度，不能与它假设独立；直接用 Young 不等式和光滑性：
\[
\mathbb E_{t-1}\|r_{t+1}\|^2
\le(1+a_t/2)\mathbb E_{t-1}\|e_t\|^2
+(1+2/a_t)L^2d_+^2\eta_t^2\mathbb E_{t-1}\|m_t\|^2.
\tag{H2}
\]
而 \(\mathbb E_{t-1}\|e_t\|^2=(1-a_t)^2\|r_t\|^2+a_t^2\sigma_t^2\)。对 \(a\le1/2\)，
\[
(1+a/2)(1-a)^2\le1-11a/8,\quad
1+2/a\le5/(2a).
\]
由 \(C\ge d_-\) 和 (C2)，
\(L^2d_+^2\eta_t^2\le a_t^2/40\)。因此额外 r 系数最多 a/8，总系数不超过 \(1-a\)；梯度平方系数最多 \(5L^2d_+^2\eta_t^2/a_t\)；噪声系数最多
\((5/4)a_t^2+(5/2)L^2d_+^2\eta_t^2a_t\le2a_t^2\)。得到 (H1)。证毕。

这个引理正面处理历史过长的问题：若任意令 \(a_t\ll\eta_t\)，漂移项 \(\eta_t^2/a_t\) 无法被以下势函数吸收。规则 \(a_t\asymp\eta_t\) 将记忆长度控制在与参数移动速度相容的尺度；它并不声称任意长的记忆总是有利。

## 2.8 定理 I：无需分母总变差界的统一势函数下降

有
\[
\boxed{\mathbb E_{t-1}\mathcal L_{t+1}
\le\mathcal L_t-{d_-\over4}\eta_t\|\mu_t\|^2
-C\eta_t\|r_t\|^2+K\eta_t^2\sigma_t^2.}
\tag{I1}
\]

**证明。** 将 (H1) 乘 \(\Lambda\) 加到 (G1)。预测误差系数满足
\(C\eta_t-\Lambda a_t\le-C\eta_t\)。梯度平方的增加量至多
\[
\Lambda 5L^2d_+^2\eta_t^2/a_t
\le (10CL^2d_+^2/c_-^2)\eta_t\le(d_-/4)\eta_t.
\]
噪声系数为 \((B\eta_t^2+2\Lambda)a_t^2\le K\eta_t^2\)。证毕。

允许 D 随历史快速变化，证明没有漏掉其变差项：这里直接分析 x 上的下降，并联合跟踪 r，且 D 实际可预测；并非照搬原 v2.1 的当前分母证明。D 的上下界保障下降，q 的具体风险改善影响实践及常数选择，却不单独决定复杂度指数。

## 2.9 定理 J：目标 \(O((1+\log T)/T^{4/5})\) 速率

令 \(\eta_t=\eta_0t^{-1/5}\)，满足 (C2)，噪声满足 (H3)。则
\[
\boxed{
\begin{aligned}
{1\over T}\sum_{t=1}^T\mathbb E\|\mu_t\|^2
\le{}&{8\mathcal L_1\over d_-\eta_0 T^{4/5}}
+{40K\sigma_0^2\eta_0\over3d_-T^{1/5}}\\
&+{432(K\upsilon)^4\eta_0^4\over d_-^4}
{1+\log T\over T^{4/5}}.
\end{aligned}}
\tag{J1}
\]
因此 \(\sigma_0=0\) 时达到请求的速率；有固定噪声底时，本日程的此界只保证慢得多的 \(O(T^{-1/5})\) 项，不能删掉。

**完整证明。** 对任意 z≥0、\(u\ge0,\delta>0\)，最大化 \(uz^{3/4}-\delta z\) 得
\[
uz^{3/4}\le\delta z+{27u^4\over256\delta^3}.
\tag{J2}
\]
令 \(u=K\upsilon\eta_t,\delta=d_-/8,z=\|\mu_t\|^2\)，在 (I1) 中吸收噪声项，得到
\[
\mathbb E_{t-1}\mathcal L_{t+1}
\le\mathcal L_t-{d_-\over8}\eta_t\|\mu_t\|^2
+K\sigma_0^2\eta_t^2+{54(K\upsilon)^4\over d_-^3}\eta_t^5.
\]
取全期望、求和、使用 \(\mathcal L_{T+1}\ge0\)：
\[
{d_-\over8}\sum_t\eta_t\mathbb E\|\mu_t\|^2
\le\mathcal L_1+K\sigma_0^2\sum_t\eta_t^2
+{54(K\upsilon)^4\over d_-^3}\sum_t\eta_t^5.
\]
因为 \(\eta_t\ge\eta_T\)、\(T\eta_T=\eta_0T^{4/5}\)、
\(\sum_{t=1}^Tt^{-2/5}\le(5/3)T^{3/5}\)、\(\sum_{t=1}^Tt^{-1}\le1+\log T\)，直接得到 (J1)。证毕。

若 \(\tau\) 独立均匀选自 \(1,\ldots,T\)，上界也就是 \(\mathbb E\|\nabla F(x_\tau)\|^2\)。由 Jensen，只得到梯度范数 \(O(\sqrt{\log T}\,T^{-2/5})\)，没有把两种指标混称为 \(T^{-4/5}\)。达到均方梯度 ≤\(\epsilon^2\) 的充分小批量梯度调用阶为 \(\widetilde O(\epsilon^{-5/2})\)；每步固定 batch 时样本调用量乘固定 batch 大小。

**已知总步数的版本。** 固定 \(\eta_t=\eta_0T^{-1/5}\)、\(a_t\in[c_-\eta_t,c_+\eta_t]\)，\(\sigma_0=0\) 时同一证明给出 \(O(T^{-4/5})\)，无需 log。log 来自无需预知 T 的递减日程，不是声称达到了不可改进的下界。

## 2.10 一般噪声、非空例子与假设敏感性

**普通有界方差。** 若只知 \(\sigma_t^2\le\bar\sigma^2\)，(I1) 给出
\[
{1\over T}\sum_t\mathbb E\|\mu_t\|^2
\le{4[\mathcal L_1+K\bar\sigma^2\sum_t\eta_t^2]\over d_-T\eta_T}.
\]
已知 T 取常步长 \(\eta\asymp T^{-1/2}\) 得 \(O(T^{-1/2})\)；未知 T 取 \(t^{-1/2}\) 得 \(O((1+\log T)/\sqrt T)\)。本方案不依靠不实的通用加速主张才能成立。

**H3 的非空、对称且非凸例子。** 取一维 \(F(x)=1-\cos x\)，随机 oracle
\[
g(x,Z)=\sin x+\sqrt{\upsilon}|\sin x|^{3/4}Z,
\quad\Pr(Z=1)=\Pr(Z=-1)=1/2.
\]
目标全局 1 光滑、下有界、非凸；噪声条件对称且方差恰为 \(\upsilon|\nabla F(x)|^{3/2}\)。这说明假设并非逻辑空集；它不是 v2.1 网络实验已经满足的模型。此 oracle 不要求每个样本函数梯度具有统一 Lipschitz 常数。

**指数为什么恰为 4/5。** 更一般地，若 \(\sigma_t^2\le\upsilon(\|\mu_t\|^2)^\theta\)、\(0\le\theta<1\)，Young 吸收产生 \(\eta_t^{1+1/(1-\theta)}\)。取 \(\eta_t\asymp t^{-(1-\theta)/(2-\theta)}\)，同一证明得到 \(\widetilde O(T^{-1/(2-\theta)})\)。\(\theta=3/4\) 正好对应 4/5。该指数是**明确附加的噪声结构**，不是从目前实验拟合出来的事实，更不是交叉项自动带来的性质。

**可预测梯度偏差。** 若 oracle 有均值偏差，则 (G1) 的无偏性步骤不成立，必须增加对应偏差项；裁剪 g 不能同时默认保持无偏。只裁剪二阶统计 v 不会破坏 g 的原始条件无偏性。

## 2.11 权重衰减与“AdamW”名称的边界

实现意义的解耦衰减可写为
\(x_{t+1}=(1-\eta_t w)x_t-\eta_tD_tm_t\)，但本章 J 定理只覆盖 w=0。
固定 w>0 的平衡一般满足 \(D\nabla F(x)+wx=0\)，不等价于 \(\nabla F(x)=0\)，也不自动等价于固定欧氏 L2 正则目标的驻点。

例如一维 \(F(x)=(x-1)^2/2\)、常 D=d>0，平衡 \(x_*=d/(d+w)\)，而 \(\nabla F(x_*)=-w/(d+w)\ne0\)。因此不能为保留名字而抹掉权重衰减条件。若希望优化明确的正则目标，可把正则梯度加入 g 并对该目标重新核验 H1–H3；那是耦合正则，不称为原样的解耦 AdamW 定理。

## 2.12 本章证明的精确强度

1. 二阶矩输入的条件风险及对称噪声收益区间是精确等式。
2. 有界分母、有限二阶统计记忆与方向跟踪拥有显式不等式保证。
3. 在消失噪声条件下，两个主要状态、一次梯度调用的候选达到目标的平均梯度平方速率。
4. 收敛证明只用预条件器的可预测有界性，不证明它比同条件下的 SGD、其他动量方法或 MARS/SOAP/Lion/Muon 有更好的指数或常数。
5. 原 v2.1、固定 beta、当前分母、固定非零权重衰减的完整组合没有被本章重新标成已满足 J 定理。
