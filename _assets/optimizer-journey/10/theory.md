# 累计矩缩减：正性、信号保留与参数位移噪声

日期：2026-11-09。研究分支，非已验证的 CrossAdamW 定稿。

本稿按用户澄清，把“累加递推”作为独立算法设计，不再将它当作 EMA 的变量代换。目标是找出最少的、能同时写清正性和累计位移噪声的改动。核心保留一个平方差缩减项与一个参数差分更新，不叠加前移梯度、下降门控或多个 EMA。

结论先行：得到了可证明的相容累计矩、固定点累计噪声界、有界凸域收敛保证，以及共享标量尺度版本的条件非凸速率；没有得到一般神经网络上比充分调参的 AdamW 更快、更稳的保证。短闭环实验也不足以支持这样的性能结论。整体属于 adaptive dual averaging 的研究分支，不能把这个已知框架宣称为首创。

## 1. 最简候选：平方差缩减 + 归一化累计状态的差分

所有平方、除法和平方根均逐坐标。每步只在当前参数 \(x_t\) 计算一次 \(g_t\)。令 \(S_0=V_0=U_0=0\)，固定 \(0\le\lambda<1\)、\(\eta>0\)，取

$$
h_t=\begin{cases}S_{t-1}/(t-1),&t>1,\\0,&t=1,\end{cases}
$$

$$\boxed{\begin{aligned}
S_t&=S_{t-1}+g_t,\\
V_t&=V_{t-1}+g_t^2-\lambda\frac{t-1}{t}(g_t-h_t)^2,\\
U_t&=\frac{S_t}{\sqrt{V_t+\varepsilon^2}},\\
x_{t+1}&=(1-\eta w\tau_t)x_t-\eta(U_t-U_{t-1}),\qquad
\tau_t=\sqrt t-\sqrt{t-1}.
\end{aligned}}\tag{1}
$$

这里 \(\varepsilon^2\) 是加在累计平方量上的固定小量，和 Adam 的根号外 epsilon 约定不同。理论中允许 \(\varepsilon=0\)，遇到全部梯度为零时定义 \(U_t=0\)；实际计算使用正 epsilon。

先分析 \(w=0\)。带权重衰减时的 \(\tau_t\) 由恒定信号匹配导出，见第 6 节；不能直接把固定强度的 AdamW 衰减接上后假设行为相同。

这个方法没有修改梯度采样位置，也不把归一化累计动量当作每一步都要再加一次的速度。它在更新**累计参数位移**。暂用 CSD（coherent cumulative shrinkage）指代该研究分支；\(\lambda=0,w=0\) 退化为无约束 AdaGrad dual averaging，而不是 AdamW。

## 2. 平方差系数不是拍脑袋：它精确匹配正性

引入仅用于推导的原始累计平方量和中心离差：

$$Q_t=\sum_{k=1}^t g_k^2,\qquad
C_t=\sum_{k=1}^t(g_k-S_t/t)^2.$$

经典在线中心离差恒等式给出

$$C_t=C_{t-1}+\frac{t-1}{t}(g_t-h_t)^2.$$

所以 (1) 精确满足

$$\boxed{
V_t=Q_t-\lambda C_t
=\frac{S_t^2}{t}+(1-\lambda)C_t.
}\tag{2}$$

从而，对任意梯度路径，

$$V_t\ge S_t^2/t,\qquad
V_t\ge(1-\lambda)Q_t,\qquad
|U_t|\le\sqrt t.\tag{3}$$

若用平均矩 \(m_t=S_t/t,\ v_t=V_t/t\) 表示，则 \(v_t\ge m_t^2\)。这里“相容”指一阶矩平方没有超过相应的二阶矩，并非 Hessian 正定性。

给定这样的缩减目标 \(Q_t-\lambda C_t\)，\((t-1)/t\) 就是精确的增量系数。它不是任意的额外超参数。固定 \(\lambda>1\) 不可能对所有路径保证正性：两步梯度 \(a,-a\) 给出 \(S_2=0,C_2=2a^2,V_2=2(1-\lambda)a^2<0\)。\(\lambda=1\) 虽然非负，却在 \(S_t=0\) 时失去尺度，且纯噪声控制退化，因此不作为稳健工作区间。

**实现方式：**维护 \(S_t,C_t\)，用 (2) 计算 \(V_t\)。这与 (1) 完全相同，但避免大数相减导致负值。只有两个持久参数规模状态，不需要保存所有梯度，也不需要第三个均值 EMA。若每步重算旧 \(U\)，可不存第三个状态；代价是多一次旧尺度计算。若缓存 \(U\)，则是三个状态，不能同时声称“两状态且无需重算”。

注意 \(V_t\ge0\) 不意味着 \(V_t\) 单调递增。例如 \(g_t=0,h_t\ne0,\lambda>0\) 时，\(V_t-V_{t-1}<0\)。

## 3. 为什么没有再给一阶累计矩强行减去一个常系数差分

可以考虑

$$A_t=A_{t-1}+g_t-\gamma(g_t-h_t),\qquad h_t=A_{t-1}/(t-1).$$

此处单独初始化 \(A_1=g_1\)，上述递推从 \(t=2\) 开始。其平均量 \(a_t=A_t/t\) 的确保持恒定信号，但递推为

$$a_t=a_{t-1}+\frac{1-\gamma}{t}(g_t-a_{t-1}).$$

令 \(\kappa=1-\gamma\)。对固定均值、独立同方差噪声，\(1/2<\kappa\le1\) 时

$$\operatorname{Var}(a_t)\sim
\frac{\kappa^2}{2\kappa-1}\frac{\sigma^2}{t}.$$

由于 \(\kappa^2/(2\kappa-1)\ge1\)，这种常系数的一阶差分缩减不优于普通样本平均的渐近方差；\(\kappa=1/2\) 为 \(\log t/t\) 量级，\(\kappa<1/2\) 更慢。偏差还以 \(t^{-\kappa}\) 衰减。

推导来自误差方差递推 \(r_t=(1-\kappa/t)^2r_{t-1}+\kappa^2\sigma^2/t^2\)；代入 \(r_t\sim A/t\) 得 \((2\kappa-1)A=\kappa^2\sigma^2\)。端点须单独处理。

因此主方案让 \(S_t\) 累加，平均量本身完成 \(1/t\) 方差缩减；不为了“多层”而再添加一个在这个模型中更差的自由度。这不是所有非平稳任务上的不可能结论。

## 4. 二阶矩的方差究竟减少了多少

本节限于固定位置 \(g_k\overset{iid}{\sim}N(\mu,\sigma^2)\)，不把沿训练轨迹变化的均值混同为固定信号。记 \(\delta=1-\lambda\)。高斯样本均值与中心离差独立，故

$$\mathbb E[V_t/t]=\mu^2+\left(\delta+\frac{\lambda}{t}\right)\sigma^2,$$

$$\boxed{
\operatorname{Var}(V_t/t)=
\frac{4\mu^2\sigma^2}{t}
+\frac{2\sigma^4}{t^2}\{1+\delta^2(t-1)\}.
}\tag{4}$$

未缩减 \(Q_t/t\) 的方差是 \(4\mu^2\sigma^2/t+2\sigma^4/t\)，因此方差精确减少

$$\frac{2\sigma^4(t-1)}{t^2}(2\lambda-\lambda^2).$$

纯噪声区的渐近方差比为 \((1-\lambda)^2\)。但估计目标也从 \(\mu^2+\sigma^2\) 改成了 \(\mu^2+\delta\sigma^2\)：这是有偏目标缩减，不能直接称为对原二阶矩更准确。更不能由 (4) 推出完整参数更新方差下降。

历史中心离差也不等于纯噪声。即便在更简单的外生漂移模型 \(g_k=\mu_k+\xi_k\) 中，若 \(\mu_k\) 是确定序列、\(\xi_k\) 独立零均值同方差，则
\(\mathbb E C_t=\sum_k(\mu_k-\bar\mu_t)^2+(t-1)\sigma^2\)。
因此该减项也会缩减真实信号的时间变化。沿自适应训练路径还需处理额外相关项；恒定信号保留不等于保留任意变化信号，这正是网络应用尚未解决的风险之一。

## 5. 参数递推真正改了什么

当 \(w=0,\eta\) 固定时，(1) 给出逐路径恒等式

$$\boxed{x_{T+1}-x_1=-\eta U_T.}\tag{5}$$

它把相关历史噪声的影响留在一个当前累计状态里。若改为 \(x_{t+1}=x_t-\eta U_t\)，就会把同一份历史信息重复积分，(5) 不再成立。

当 \(g_t\equiv\mu\ne0,\varepsilon=0\) 时，\(C_t=0,V_t=t\mu^2\)，所以

$$U_t=\sqrt t\,\operatorname{sign}(\mu),\qquad
x_{t+1}-x_t=-\eta\tau_t\operatorname{sign}(\mu).$$

总下降位移为 \(-\eta\sqrt T\,\operatorname{sign}(\mu)\)，不会因为中心化而消失。单步响应约为 \(-\eta\operatorname{sign}(\mu)/(2\sqrt t)\)，并非恒定步长。

对固定 iid 有限二阶矩且 \(\mu\ne0\)，大数律还给出

$$\frac{U_t}{\sqrt t}\longrightarrow
\frac{\mu}{\sqrt{\mu^2+(1-\lambda)\sigma^2}}.$$

这说明缩减增强了持久信号的响应，但这是固定信号模型的响应，不是沿非凸训练路径的加速定理。

**与已有框架的关系：**(5) 可写成

$$x_{t+1}=\arg\min_x\left\{
\langle S_t,x-x_1\rangle+
\frac{1}{2\eta}\sum_j\sqrt{V_{t,j}+\varepsilon^2}(x_j-x_{1,j})^2
\right\}.$$

因此它是自适应双重平均。创新审查必须针对具体缩减正则器及其分析，不能把“参数更新写成差分”本身作为新范式。

## 6. 完整累计位移噪声：包含时间相关项的界

假设某一坐标输入是独立、关于零对称的随机变量 \(g_t=\xi_t\)。这是固定位置的纯噪声检验；不是假定自适应训练轨迹上的完整梯度也满足此条件。令 \(Q_T>0\) 几乎处处。

独立对称符号使

$$\mathbb E\left[\frac{S_T^2}{Q_T}\right]=1.$$

证明：对绝对值条件化，交叉符号项的条件期望为零，对角项和为 \(Q_T\)。若允许全部为零的事件，定义该比值为零，期望至多为 1。

结合 (3) 和 (5)，

$$\boxed{
\mathbb E|x_{T+1}-x_1|^2
\le\frac{\eta^2}{1-\lambda}
}\tag{6}
$$

对所有 \(T\) 成立。各坐标无需相互独立，总范数界为 \(d\eta^2/(1-\lambda)\)。

这里衡量的是
\(\mathbb E|\sum_t\Delta x_t|^2\)，包含不同时刻更新之间的协方差；它不是 \(\sum_t\mathbb E|\Delta x_t|^2\)，不是路径总长度，也不是最大单步幅度。净累计位移有界不代表路径已经收敛。在 iid 高斯噪声下，\(U_t\) 的分布极限为 \(N(0,1/(1-\lambda))\)，均方并不趋零，不能把 (6) 宣称为收敛到精确点。

### 与固定线性动量的比较边界

若 \(d_t=\sum_{j\ge0}a_j\xi_{t-j}\) 是稳定的固定线性滤波器，\(\sum_j(1+j)|a_j|<\infty\)，直流增益 \(H=\sum_ja_j\ne0\)，则

$$\operatorname{Var}\left(\sum_{t=1}^T d_t\right)
=\sigma^2H^2T+O(1).$$

这是通过累计协方差或滤波卷积直接得到的。它解释了为什么只换一个保留信号的固定 EMA 很难消除累计噪声增长。(1) 使用时变累计状态，避开了该固定滤波器条件。该结论不适用于所有自适应 AdamW，不能据此声称一般 AdamW 的位移一定随机游走。

### 为什么权重衰减也要匹配

令 \(a_t=1-\eta w\tau_t\)，要求 \(0\le a_t\le1\)。常量 \(w\ge0\) 时，\(\eta w\le1\) 是充分条件。恒定非零信号下

$$x_{t+1}=(1-\eta w\tau_t)x_t
-\eta\tau_t\operatorname{sign}(\mu),$$

即下降与解耦衰减使用同一衰减时标。若直接使用 \(1-\eta w\) 而下降项约为 \(t^{-1/2}\)，固定衰减会逐渐压倒恒定信号输入。

写 \(P_T=\prod_{k=1}^T a_k\)，离散分部求和给出

$$x_{T+1}-P_Tx_1
=-\eta\left[U_T-\sum_{t=1}^{T-1}b_{T,t}U_t\right],$$

$$b_{T,t}=(1-a_{t+1})\prod_{k=t+2}^T a_k\ge0,\qquad
\sum_{t=1}^{T-1}b_{T,t}=1-\prod_{k=2}^T a_k\le1.$$

对同样的外生纯噪声模型，Minkowski 不等式和 (6) 给出

$$\mathbb E|x_{T+1}-P_Tx_1|^2\le
\frac{4\eta^2}{1-\lambda}.$$

这是匹配衰减版本的界，不是一般随机非凸收敛证明。该衰减日程的优越性也尚未通过网络训练验证。

若 \(\eta_t\) 随时间变化，要保持望远镜结构，应把优化项写成
\(-(\eta_tU_t-\eta_{t-1}U_{t-1})\)，不能直接用
\(-\eta_t(U_t-U_{t-1})\) 后沿用 (5)。

## 7. 非单调二阶矩仍可控制总变差

记 \(D_t=\sqrt{V_t+\varepsilon^2}\)、\(\delta=1-\lambda>0\)。即使 \(V_t\) 会下降，增量仍有

$$V_t-V_{t-1}\le g_t^2.$$

因此

$$\begin{aligned}
P_T^D&:=\sum_{t=1}^T(D_t-D_{t-1})_+
\le\sum_{t=1}^T\frac{g_t^2}{D_t}\\
&\le\frac1{\sqrt\delta}\sum_{t=1}^T\frac{g_t^2}{\sqrt{Q_t}}
\le\frac{2\sqrt{Q_T}}{\sqrt\delta}.
\end{aligned}\tag{7}$$

最后一步由
\(g_t^2/\sqrt{Q_t}\le2(\sqrt{Q_t}-\sqrt{Q_{t-1}})\)
求和得到，零项按零处理。由于 \(D_T\ge D_0\)，负变差
\(N_T^D:=\sum(D_{t-1}-D_t)_+\le P_T^D\)，故总变差至多 \(4\sqrt{Q_T/\delta}\)。

这提供了一个无需 running-max、无需额外分母门控的证明入口。它控制的是累计变化量，不保证每步比 AdamW 的尺度更平滑。

### 一个可以完整证明的凸保证

本段针对明确的**有界凸域版本**，不偷换成无约束神经网络保证。设
\(K=\{x:|x_j-x_{1,j}|\le B_j\}\)，每轮 \(f_t\) 凸，\(g_t\in\partial f_t(x_t)\)，\(|g_{t,j}|\le G_j\)，取 \(\varepsilon_j^2=\delta G_j^2>0\)、\(w=0\)，更新

$$x_{t+1}=\Pi_K(x_1-\eta S_t/D_t).$$

这里投影的是锚定累计解，不能擅自换成对单步差分更新的投影。没有投影且迭代点一直在同一个这样的域内时，同一证明也适用。

对任何 \(x^*\in K\)，有

$$
\boxed{\begin{aligned}
\sum_{t=1}^T[f_t(x_t)-f_t(x^*)]
\le{}&\sum_j\frac{D_{T,j}(x_j^*-x_{1,j})^2}{2\eta}\\
&+\frac{\eta}{2}\sum_{t,j}\frac{g_{t,j}^2}{D_{t-1,j}}
+\sum_j\frac{B_j^2}{2\eta}N_{T,j}^D.
\end{aligned}}\tag{8}
$$

**证明。** 平移 \(x_1=0\)。逐坐标定义
\(\Phi_D(S)=\max_{|z|\le B}\{-Sz-Dz^2/(2\eta)\}\)。
其关于 \(S\) 的导数是负的当前 DA 解，导数 Lipschitz 常数至多 \(\eta/D\)。因此

$$\Phi_{D_{t-1}}(S_t)-\Phi_{D_{t-1}}(S_{t-1})
\le-g_tx_t+\frac{\eta g_t^2}{2D_{t-1}}.$$

改变尺度的附加量满足

$$\Phi_{D_t}(S_t)-\Phi_{D_{t-1}}(S_t)
\le\frac{B^2}{2\eta}(D_{t-1}-D_t)_+.$$

相加、求和，并用
\(\Phi_{D_T}(S_T)\ge-S_Tx^*-D_T(x^*)^2/(2\eta)\)
即得线性 regret 界。凸性把函数值差界在该线性 regret 内。

又因为 \(D_{t-1}^2\ge\delta(Q_{t-1}+G^2)\ge\delta Q_t\)，(7) 和 (8) 给出

$$R_T\le
\sum_j\left[
\frac{\sqrt{Q_{T,j}+\delta G_j^2}(x_j^*-x_{1,j})^2}{2\eta}
+\frac{\eta\sqrt{Q_{T,j}}}{\sqrt\delta}
+\frac{B_j^2\sqrt{Q_{T,j}}}{\eta\sqrt\delta}\right]
=O(\sqrt T).$$

对固定凸目标、条件无偏随机梯度，通过在线到批量转换得到平均迭代点的 \(O(T^{-1/2})\) 期望函数值差。该速率不快于经典随机凸最优阶；此处价值是说明缩减导致的非单调尺度不必破坏该阶。不能把它写成非凸加速或全新通用 DA 收敛理论。

### 共享标量尺度的简化版：一个条件非凸定理

下面改为**整个向量共享一个标量 \(V_t\)**：

$$S_t=S_{t-1}+g_t,\qquad
V_t=V_{t-1}+\|g_t\|^2-\lambda\frac{t-1}{t}
\left\|g_t-\frac{S_{t-1}}{t-1}\right\|^2,$$

其中第一步的差分减项定义为零，\(D_t=\sqrt{V_t+\varepsilon^2}\)，\(w=0\)，
\(x_{t+1}=x_1-\eta S_t/D_t\)。这需要一个向量状态 \(S\) 与一个标量中心离差状态。它的尺度适应性弱于逐坐标 AdamW；下面的证明不能直接推广到逐坐标版本。

设 \(F\) 为 \(L\)-光滑、下界 \(F_*\) 有限；\(g_t=\nabla F(x_t)+\xi_t\) 条件无偏且
\(\mathbb E[\|\xi_t\|^2\mid\mathcal F_{t-1}]\le\sigma^2\)。
再明确假设存在与 \(T\) 无关的确定常数 \(B,M,G\)，使所有迭代点几乎处处满足

$$\|x_t-x_1\|\le B,\quad
0\le F(x_t)-F_*\le M,\quad
\|\nabla F(x_t)\|\le G.$$

这些是额外的轨迹有界条件，算法本身尚未保证；如果只在每个有限 \(T\) 事后取越来越大的 \(B_T\)，就不能宣称以下常数阶速率。

定义

$$K_\delta=\max\{1,\delta^{-1/2}-1\},\quad
A_0=\frac{\varepsilon[F(x_1)-F_*]}{\eta},$$

$$A_1=\frac{4(M+GB)}{\eta\sqrt\delta}
+\frac{2L\eta}{\sqrt\delta}
+\frac{4LB^2K_\delta}{\eta\sqrt\delta}.$$

则

$$\boxed{
\frac1T\sum_{t=1}^T\mathbb E\|\nabla F(x_t)\|^2
\le\frac{2A_0+A_1^2}{T}+\frac{2A_1\sigma}{\sqrt T}.
}\tag{9}$$

**推导。** 令 \(y_t=x_t-x_1,\ b_t=D_t/\eta\)，完整参数递推等价于

$$g_t=-b_t(x_{t+1}-x_t)-(b_t-b_{t-1})y_t.$$

光滑性给出

$$\langle\nabla F(x_t),g_t\rangle
\le b_t[F(x_t)-F(x_{t+1})]
+\frac L2 b_t\|x_{t+1}-x_t\|^2
-(b_t-b_{t-1})\langle\nabla F(x_t),y_t\rangle.$$

在这个标量版本，(2)、(7) 将平方替换为范数平方后仍成立，记
\(Q_T=\sum_t\|g_t\|^2\)，故
\(\mathrm{TV}(D)\le4\sqrt{Q_T/\delta}\)。
又因 \(D_{t-1}^2\le Q_{t-1}+\varepsilon^2\le Q_t+\varepsilon^2\)
和 \(D_t^2\ge\delta Q_t+\varepsilon^2\)，有
\(D_{t-1}/D_t\le\delta^{-1/2}\)，从而
\(|D_t-D_{t-1}|/D_t\le K_\delta\)。

于是完整更新的能量满足

$$\begin{aligned}
\sum_t b_t\|x_{t+1}-x_t\|^2
&\le2\eta\sum_t\frac{\|g_t\|^2}{D_t}
+\frac{2B^2}{\eta}\sum_t\frac{(D_t-D_{t-1})^2}{D_t}\\
&\le\frac{4\eta\sqrt{Q_T}}{\sqrt\delta}
+\frac{8B^2K_\delta\sqrt{Q_T}}{\eta\sqrt\delta}.
\end{aligned}$$

函数值项离散分部求和，末项 \(-b_T[F(x_{T+1})-F_*]\le0\)，其余变差项分别由 \(M\) 与 \(GB\) 控制，得逐路径不等式

$$\sum_t\langle\nabla F(x_t),g_t\rangle
\le A_0+A_1\sqrt{Q_T}.$$

设 \(H_T=\sum_t\mathbb E\|\nabla F(x_t)\|^2\)。条件无偏性及方差界给出
\(\mathbb E Q_T\le H_T+T\sigma^2\)，故

$$H_T\le A_0+A_1\sqrt{H_T+T\sigma^2}
\le A_0+\tfrac12H_T+\tfrac12A_1^2+A_1\sigma\sqrt T.$$

整理即得 (9)。该证明没有把当前随机尺度当成独立于 \(g_t\)，而是先建立完整路径上的不等式再取期望。

这表明共享尺度分支在这些条件下可以进入越来越小的平均梯度平方区间，但没有说明最后一次迭代收敛，也没有证明神经网络实际满足轨迹有界条件。它与已有 ADA-DA 非凸分析有直接关系；这里需要审查的具体差别是允许中心离差缩减后尺度非单调，并显式支付其变差代价。该界随 \(\lambda\) 增大通常更松，不是相对 AdamW 的加速证明。

## 8. 其他简单构造的探索结果

### A. 直接反复使用累计均值归一化

保持同样 \(S,V\)，但令
\(x_{t+1}=x_t-\eta S_t/\sqrt{tV_t}\)。
恒定梯度下保持常量步幅，但累计信息被反复使用；短闭环实验的误差明显更大。它不能继承 (5)–(6)。

### B. 再缩减一阶累计状态

尝试过无新超参数的有理缩减

$$r_t=\frac{S_t^2}{S_t^2+\lambda C_t/(1-\lambda)},\qquad
U_t^{\rm joint}=r_tS_t/\sqrt{V_t+\varepsilon^2}.$$

零除定义为零。它仍保留恒定非零信号，并保证 \(V_t\ge(r_tS_t)^2/t\)。纯高斯固定点下，大样本极限为

$$U_t^{\rm joint}\Rightarrow
\frac{Z^3}{\sqrt{1-\lambda}\,[Z^2+\lambda/(1-\lambda)]},
\quad Z\sim N(0,1).$$

\(\lambda=1/2\) 时，其极限二阶矩为
\(2\mathbb E[Z^6/(1+Z^2)^2]\approx0.934077\)，小于未缩减 DA 的 1，也远小于仅缩减分母的 2。这是固定点分布计算，不是有限时间所有噪声分布下的支配定理。

但**闭环二次实验并未因此改善**。抑制固定点噪声同时改变了信号反馈灵敏度，因此不把该一阶缩减加入主方案，也不把孤立的 0.934077 当作训练优势。信噪比式缩减本身还有已有统计估计与优化近邻，不能仅凭这个比例宣称首创。

### C. 缩减极限 \(\lambda=1\)

当 \(\varepsilon=0\) 时，\(V_t=S_t^2/t\)、\(U_t=\sqrt t\,\operatorname{sign}(S_t)\)，可能因为累计和过零产生大的跳变。在高斯纯噪声下 \(\mathbb E U_t^2=t\)，(6) 的统一有界性消失。这给出了“缩得越多越好”的直接反例。

## 9. 本次实际验证与性能边界

代码与原始数据位于 [实验目录](../experiments/20261108_cumulative/README_ZH.md)。只做 CPU 数学与标量机制检验，没有使用租用 GPU，也没有新增神经网络训练。

- 符号恒等式、正性、累计位移恒等式、匹配衰减展开及凸 regret 数值检查通过；浮点恒等误差约 \(10^{-12}\) 或更小。
- 固定点纯噪声：20,000 条独立高斯路径，至 2,048 步。未缩减 DA 的 \(\mathbb E U_T^2\) 估计 0.98381；仅二阶缩减 \(\lambda=.5\) 为 1.96477，与上界 1 和 2 相容。有限样本估计不是精确期望。
- 闭环：\(F(x)=kx^2/2,\ g=kx+N(0,1)\)，\(x_1=3\)，\(k=1,10\)，每种配置 512 条配对噪声路径、4,096 步、6 点学习率搜索。共 10 方法 × 6 学习率 × 2 曲率 = 120 个配置；全部保留，不删除失败结果。
- 同时加入 \(t^{-1/2}\) 学习率衰减的 Adam 基线，避免把任何衰减步长的收益误认作新算法独占贡献。

以 \(k=1\)、每方法选择前 300 步平均 MSE 最小的网格点为例：

|方法|选中 \(\eta\)|前 300 步平均 MSE|末 1,024 步平均 MSE|尾部更新 RMS|
|---|---:|---:|---:|---:|
|Adam，固定学习率|0.3|0.230689|0.141016|0.064537|
|Adam，学习率乘 \(t^{-1/2}\)|1.0|0.084774|0.008361|0.003832|
|未缩减 DA|1.0|0.111283|0.009000|0.016615|
|CSD，\(\lambda=.5\)|1.0|0.104340|0.012153|0.023459|
|联合一阶缩减，\(\lambda=.5\)|1.0|0.121396|0.013480|0.025817|

所有这些运行 \(w=0\)，因此表中的 Adam 等于零权重衰减的 AdamW。选择和评估使用同一批模拟路径，属于探索性调参展示，不是独立保留集排名。不能据此作统计显著性或普适优越性声称。6 个相同数值的学习率提供相等搜索点数，也不保证各算法最佳尺度都被同等覆盖。

存在局部正例：\(k=1,\eta=.3\) 时，CSD 相对 DA 的前段 MSE 从 0.781379 降到 0.626603，尾段从 0.009054 降到 0.006705；但充分调参后的独立收益仍不成立，且 \(k=10\) 的结果也不支持普遍优势。不能只挑这个正例写摘要。

这次没有增加长训练，因为核心对照已经否定了“现有简式可直接宣称同时更快更稳”。矩估计方差降低与完整闭环误差降低之间仍有真实缺口。

## 10. 新颖性：哪些可以主张，哪些已经有前人

检索截至 2026-11-09，使用原论文，而非仅依赖综述或标题。相关查询包括 cumulative/centered second moment shrinkage、dual averaging variance reduction、AdaMomentum、AdamS、nonconvex dual averaging、optimistic dual averaging。有限检索无法证明所有论文均不存在相同公式。

|原始文献|直接相关部分|对本分支的约束|
|---|---|---|
|[Duchi–Hazan–Singer, AdaGrad, JMLR 2011](https://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)|自适应二次正则化与 dual averaging|累计梯度除累计平方根、锚定参数更新是已有框架|
|[Defazio–Jelassi, MADGRAD, JMLR 2022](https://jmlr.org/papers/volume23/21-0226/21-0226.pdf)|加权累计矩、三次根预条件器、原变量平均|不能把累计矩与神经网络训练的结合称为首创；其具体尺度和平均结构不同|
|[Huang–Lee, RAMDA, 2024](https://arxiv.org/html/2403.14398v2)|同时对梯度估计与预条件器进行 dual averaging，并研究方差消失和结构识别|“一阶二阶同时方差缩减 + DA”这一宽泛概念已有直接前人|
|[Wang 等, AdaMomentum](https://arxiv.org/html/2106.11514v3)|一阶 EMA 的平方进入第二层 EMA|不能把用平滑信号替代原始梯度平方作为新概念；本稿中心离差分解不同|
|[AdamS, 2025](https://arxiv.org/html/2505.16363v1)|归一化量为 \(\beta_2m_{t-1}^2+(1-\beta_2)g_t^2\)|平滑一阶信号与当前梯度平方混合已有；本稿不是该单状态方案|
|[Liu–Saad–Kotłowski–Orabona, 2025](https://arxiv.org/html/2505.21394v1)|随机非凸 DA 收敛与 ADA-DA|不能宣称首次证明 DA 非凸收敛；本稿非单调缩减尺度也不能直接继承其定理|
|[Pethick 等, SODA, 2026](https://arxiv.org/html/2605.11172v1)|以 optimistic DA 统一多种优化器，并提出 \(1/k\) 权重衰减日程|参数递推与衰减日程的联合设计已有宽泛先例；具体 \(\sqrt t-\sqrt{t-1}\) 匹配需公式级比较|

在线中心离差更新也是经典 Welford 型恒等式，(2) 中的均值平方与样本离差分解不是新统计定理。(8) 使用标准凸共轭势函数分析；不能把证明工具本身称作首创。

**当前可以负责任地作为研究贡献候选的窄命题：**

> 对仅含一个中心平方差减项的累计自适应矩，联合给出矩相容性、缩减后非单调尺度的总变差控制、完整累计参数位移噪声界，以及匹配的解耦衰减结构；明确刻画二阶矩方差下降与闭环优化噪声之间的差异。

在已核对来源中，尚未定位到与 (1)–(2) 及该整组分析逐项相同的公式。但“未找到完全相同”只支持继续查新，不能证明首创；算法又与已知 DA 高度相关，当前不足以用“全新优化范式”包装。

要形成有竞争力的 ICML 主张，缺的是缩减本身带来的不可被学习率重调解释的优势、一般训练轨迹上的完整误差分析，以及真实任务证据。现在最有根据的是这组简洁结构与边界定理，不是已经胜过 AdamW 的结论。
