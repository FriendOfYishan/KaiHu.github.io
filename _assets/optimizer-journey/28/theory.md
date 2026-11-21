# 适合AWA的修正应解决什么：误差递推、低成本候选与等价性审计

2026-11-11。研究笔记。原Joint-AWA主候选不变；本稿没有新增训练结果，不认定候选原创性或实际优势。

## 1. 修正估计器是一个方向，但目标不是复制外形

记mu_t=grad F(x_t)、delta_t=mu_t-mu_{t-1}。在当前梯度抽样前，选择一个由历史决定的修正c_t，考虑
\[
m_t=(1-\alpha_t)(m_{t-1}+c_t)+\alpha_tg_t.
\]
则精确地
\[
e_t=(1-\alpha_t)(e_{t-1}+c_t-\delta_t)+\alpha_t\xi_t.
\]
由于c_t与x_t在本步抽样前已知，当前噪声的条件交叉项为0。Young不等式给出
\[
\boxed{E\|e_t\|^2\le(1-\alpha_t)E\|e_{t-1}\|^2
+\frac{(1-\alpha_t)^2}{\alpha_t}E\|c_t-\delta_t\|^2
+\alpha_t^2\sigma^2.}
\]
因此真正需要证明的是：在允许较大eta时，修正后的移动残差E||c-delta||^2比原来E||delta||^2更小。如果只减小常数，通常没有改善幂指数；若要宣称更快复杂度，必须证明残差衰减阶数或与下降耗散的耦合得到改善，并完成完整下降分析。

c_t若直接使用当前g_t，上述零交叉项一般失效，需要重新推导。这是采用同样本修正时不能省略的一步。

## 2. 一个从移动残差直接推出的低成本候选

原更新给出x_t-x_{t-1}=-eta_{t-1}D_{t-1}m_{t-1}。若F二阶连续，则
\[
\delta_t=-\eta_{t-1}\bar H_tD_{t-1}m_{t-1},
\quad \bar H_t=\int_0^1\nabla^2F(x_{t-1}+s(x_t-x_{t-1}))ds.
\]
探索用可预测的标量kappa_t近似预条件曲率的作用：
\[
c_t=-\kappa_t\eta_{t-1}m_{t-1}.
\]
于是得到候选
\[
\boxed{
m_t=(1-\alpha_t)(1-\kappa_t\eta_{t-1})m_{t-1}+\alpha_tg_t,
\quad v_t=(1-\alpha_t)v_{t-1}+\alpha_tg_t^2,
\quad x_{t+1}=x_t-\eta_t\frac{m_t}{\sqrt{v_t}+\varepsilon}.
}
\]
首步alpha_1=1，无需eta_0的前一步解释。此处eta_0若用于学习率尺度，与eta_{t-1}的下标需要在实现中区分。

这里只修改一阶矩的标量系数；若kappa本身可廉价获得，则没有额外梯度或新的向量状态，也无需新增逐坐标乘加数目。但没有证明kappa可以廉价可靠估计。

残差精确为
\[
c_t-\delta_t=\eta_{t-1}(\bar H_tD_{t-1}-\kappa_t I)m_{t-1}.
\]
在预条件曲率作用接近一个标量的局部模型中，它可能有帮助；一般非凸问题中的负曲率、各向异性或曲率失配可能使它更差。v是梯度二阶矩，不是Hessian，不能直接用v宣称已识别曲率。

如果0<=kappa_t eta_{t-1}<=1，则beta_t=(1-alpha_t)(1-kappa_t eta_{t-1})满足0<=beta_t<=1-alpha_t。递推可视作旧动量、新梯度和零向量的凸组合，因此
\[
m_t^2\le\beta_t m_{t-1}^2+\alpha_tg_t^2
\le(1-\alpha_t)v_{t-1}+\alpha_tg_t^2=v_t.
\]
这保留了方向有界性。但第26版依赖原一阶系数的q与能量恒等式，需要为新beta重新证明，不能直接继承全部定理。

## 3. 等价性审计：不能仅凭交叉系数宣布原创

定义标量总权重z_0=0，z_t=beta_t z_{t-1}+alpha_t；令bar m_t=m_t/z_t。则
\[
\bar m_t=(1-\tilde\alpha_t)\bar m_{t-1}+\tilde\alpha_tg_t,
\quad\tilde\alpha_t=\alpha_t/z_t,
\]
\[
x_{t+1}=x_t-(\eta_tz_t)\frac{\bar m_t}{\sqrt{v_t}+\varepsilon}.
\]
因此上述形式可精确重写成另一种一阶记忆日程与有效学习率日程；v仍使用原alpha。这不是“完全等于原Joint”的说法，但说明它没有凭空产生新的梯度信息。

若仅提出该公式，没有kappa选择原则、可验证的残差改善和相对重新调日程的优势，就不足以构成原创算法贡献。它目前只作为推导出的候选及审计示例，不替换主候选。

## 4. 均值方差缩减与二阶矩缩减必须区分

降低Var(m)可以是合理目标，但必须同时控制其相对mu的偏差。标量收缩估计hat mu=lambda g的均方误差为
\[
E(\lambda g-\mu)^2=(1-\lambda)^2\mu^2+\lambda^2\sigma^2.
\]
方差变小并不保证均方误差更小。将v减小也不是在减少随机梯度噪声，而是在改变预条件器；还可能破坏m^2<=v、放大更新。

因此现阶段不建议恢复任意二阶矩减法。优先研究均值估计器的移动残差，并保留未修正梯度平方的v作为明确基线；后续是否改变v须单独证明与消融。

## 5. 理论弱不意味着大模型实测一定弱

当前Joint的加权梯度平方O(T^(-1/3))及幂律平衡后的O(log T/T^(2/5))都是特定强假设下的保守保证。Adam/AdamW文献使用不同输出指标、正则化目标及参数条件，不能无条件只按指数排名。

最坏情形的较弱上界不排除在某类模型上的更好有限预算表现，但也不提供这种表现的证据。Muon的相关工作提供了大模型训练与计算效率的实证案例；它利用矩阵结构，不能由此推断AWA共享同样优势。

当前AWA相关实验主要是小任务与低维诊断，还没有语言模型预训练证据。比继续构造理想曲率控制器更直接的实证路线，是冻结原Joint，先进行小型Transformer预训练的同预算比较，报告验证损失对token、FLOPs和墙钟时间的关系；使用同等调参预算，并明确权重衰减与当前无衰减理论的区别。

参考：[MARS原文](https://arxiv.org/html/2411.10438v4)、[Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)、[Practical Efficiency of Muon for Pretraining](https://arxiv.org/abs/2505.02222)。这些来源用于比较研究路线，不证明本候选有效。
