# 1. v2.1 正式理论及非平稳缺口修补

本章正式化原 v2.1，不把新候选的改变归给旧算法。基础来源是 `../../Shrink AdamW/v2.1/theory/THEORY_V21_ZH.md` 与 `THEORY_CANDIDATE_NONCONVEX_ZH.md`。本章保留其正确结论，新增非平稳条件递推及其适用边界。

## 1.1 定义与信息顺序

\(\mathcal F_{t-1}\) 为第 t 次梯度采样前的信息，\(x_t,h_t,\alpha_t\) 均可测。逐坐标写

\[
g_t=\mu_t+\xi_t,\quad \mu_t=\nabla_iF(x_t),\quad
\mathbb E_{t-1}\xi_t=0,
\]
\[
s_t=\mathbb E_{t-1}\xi_t^2,\quad k_t=\mathbb E_{t-1}\xi_t^3,
\quad M_{4,t}=\mathbb E_{t-1}\xi_t^4,
\quad e_t=h_t-\mu_t.
\]

原输入、目标和更新为

\[
q_t^{21}=g_t^2-2\alpha_t h_t(g_t-h_t),\quad
\nu_t=\mu_t^2+s_t,
\tag{1}
\]
\[
h_t=\widehat m_{t-1},\quad m_t=p m_{t-1}+(1-p)g_t,
\quad v_t=bv_{t-1}+(1-b)q_t^{21},
\]
\[
x_{t+1}=(1-\eta_t w)x_t-
\eta_t\widehat m_t/(\sqrt{\widehat v_t}+\varepsilon).
\tag{2}
\]

初始 \(m_0=v_0=0,h_1=0\)，偏差修正为 \(1-p^t,1-b^t\)。除特别声明外 \(\alpha_t\in[0,1]\)。本章平方和除法逐坐标；向量范数为欧氏范数。

## 1.2 定理 A：非负性与精确条件风险

省略 t，令 \(W=M_4-s^2\)、\(C=2\mu s+k\)、\(V=4\mu^2s+4\mu k+W\)。有

\[
q^{21}=(g-\alpha h)^2+\alpha(2-\alpha)h^2
=(1-\alpha/2)g^2+2\alpha(h-g/2)^2.
\tag{3}
\]

因此对所有实数 g,h 保证非负的精确区间为 \([0,2]\)，在算法区间 \([0,1]\) 有 \(q^{21}\ge g^2/2\)。若 \(|g|,|h|\le G\)，则 \(q^{21}\le5G^2\)。

令 \(P=hC,Q=h^2(s+e^2)\)。条件 MSE 为

\[
R_{21}(\alpha)=V-4P\alpha+4Q\alpha^2,
\quad \mathbb E q^{21}-\nu=2\alpha he,
\tag{4}
\]
\[
\operatorname{Var}(q^{21})=4(\mu-\alpha h)^2s+
4(\mu-\alpha h)k+W.
\tag{5}
\]

当 \(Q>0\)，\(\alpha_*=[P/(2Q)]_{[0,1]}\)；\(Q=0\) 时修正几乎必然为零，取 0。

**证明。** (3) 直接展开；\(\alpha<0\) 取 \(g=0,h=1\)，\(\alpha>2\) 取 \(g=\alpha,h=1\)，得到负值。再由
\[
q^{21}-\nu=2\alpha he+2(\mu-\alpha h)\xi+(\xi^2-s)
\]
平方取条件期望即得 (4)–(5)；最小化一元凸二次函数得到系数。证毕。

**同目标比较。** V2 的 \(q^2=(1-\lambda)g^2+\lambda h^2\) 对 \(\nu\) 的风险是
\[
R_2=(1-\lambda)^2V+\lambda^2(h^2-\nu)^2.
\]
最优风险为 \(V(h^2-\nu)^2/[V+(h^2-\nu)^2]\)，零分母时为零。两方法必须对同一个目标比较。取 \(h=\mu=1\)、标准高斯噪声，原 v2.1 最优风险为 2，而 V2 最优风险为 \(6/7\)，已否定无条件单步支配。

## 1.3 定理 B：对称性消除什么

若 \(h=\mu,k=0,\alpha=1\)，则
\[
q^{21}=\mu^2+\xi^2,\quad
\operatorname{MSE}(q^{21},\nu)=W,
\quad \operatorname{MSE}(g^2,\nu)-W=4\mu^2s.
\tag{6}
\]
证明由 (4) 代入。仅需零三阶矩，不必假设完整分布对称。

但 \(g\) 的方向噪声仍是 \(\xi\)，方差仍为 s。消掉的是**二阶矩输入的信号—噪声交叉项**，不是免费消掉随机梯度噪声。预测错误时
\[
R_{21}(1)=W-4ek+4e^2[s+(\mu+e)^2].
\tag{7}
\]
零三阶矩使额外风险从 e 的二阶起始；不保证任意大小的历史误差仍有利。

## 1.4 定理 C：历史相关性完整保留的稳态公式

本节仅考虑固定 x、双向 iid 创新、有限四阶矩、确定常系数。令 \(p,b\in[0,1)\)、\(d_1=1-p\)，
\[
e_t=d_1\sum_{j\ge1}p^{j-1}\xi_{t-j},\quad
H=\frac{1-p}{1+p}s,\quad K_e=\frac{d_1^3}{1-p^3}k,
\]
\[
J_\xi=M_4-s^2,\quad
J_e=2H^2+(M_4-3s^2)\frac{d_1^4}{1-p^4}.
\]
对通用输入
\[
q_t=\mu^2+2\mu(A\xi_t+B e_t)+U\xi_t^2+J e_t\xi_t+N e_t^2
\tag{8}
\]
定义
\[
\Gamma_0=4\mu^2(A^2s+B^2H)+U^2J_\xi+J^2Hs+N^2J_e
+4\mu(AUk+BNK_e),
\]
\[
K_1=4\mu^2B(pBH+d_1As)+2\mu B(pNK_e+d_1Uk),
\]
\[
K_2=2\mu N(p^2BK_e+d_1^2Ak)
+N(p^2NJ_e+d_1^2UJ_\xi+2pd_1JHs).
\]
则稳态 \(\bar v_t=(1-b)\sum_{j\ge0}b^jq_{t-j}\) 满足
\[
\boxed{\mathbb E(\bar v_t-\nu)^2=
[(U-1)s+NH]^2+
\frac{1-b}{1+b}\left(\Gamma_0+
\frac{2bK_1}{1-pb}+\frac{2bK_2}{1-p^2b}\right).}
\tag{9}
\]

参数对应如下：

|输入|A|B|U|J|N|
|---|---|---|---|---|---|
|原 v2.1|\(1-\alpha\)|\(\alpha\)|1|\(-2\alpha\)|\(2\alpha\)|
|V2|\(1-\lambda\)|\(\lambda\)|\(1-\lambda\)|0|\(\lambda\)|
|新候选的嵌套输入|\(\kappa(1-\alpha)\)|\(1-\kappa+\kappa\alpha\)|\(\kappa\)|\(-2\kappa\alpha\)|\(1-\kappa+2\kappa\alpha\)|

**证明。** 独立创新的二、三阶矩仅保留同一创新项；四阶矩保留配对项与四阶累积量，得到 H、\(K_e,J_e\)。当前 \(\xi_t\) 与 \(e_t\) 独立，展开得均值 \(\mu^2+Us+NH\) 与方差 \(\Gamma_0\)。对滞后 \(\ell\ge1\)，写
\[
e_t=p^\ell e_{t-\ell}+d_1p^{\ell-1}\xi_{t-\ell}+Z,
\]
其中 Z 与两项独立、均值零。对当前创新和 Z 条件平均后，与 \(q_{t-\ell}\) 求协方差仅剩一次与二次单项式，逐项配对得到
\[
\operatorname{Cov}(q_t,q_{t-\ell})=K_1p^{\ell-1}+K_2p^{2(\ell-1)}.
\]
归并二阶 EMA 双重和中的同滞后项：
\[
\operatorname{Var}(\bar v)=\frac{1-b}{1+b}
[\Gamma_0+2\sum_{\ell\ge1}b^\ell\operatorname{Cov}(q_t,q_{t-\ell})].
\]
求两个几何级数并加偏差平方即得 (9)。有限四阶矩与几何衰减保证换序合法。p=0 时只保留滞后 1。证毕。

高斯、\(p=.9,b=.999\) 时，对两种方法分别优化确定常系数，v2.1 胜出当且仅当
\[
\mu^2/s>872239446817/72673232270\simeq12.0022107119.
\]
这不是普遍阈值。变动 x、非高斯矩或数据依赖系数不能直接使用此特例。零初值偏差修正过程趋近稳态，可通过相同创新耦合、预测器尾项在 \(L^4\) 的几何衰减及稳定 EMA 在 \(L^2\) 的连续性证明。

## 1.5 新增定理 D：非平稳、可预测自适应遗忘的精确递推

设 \(q_t=\nu_t+b_t+\zeta_t\)，其中 \(\nu_t,b_t\) 为历史可测，\(\mathbb E_{t-1}\zeta_t=0\)，条件方差为 \(S_t\)。采用历史可测 \(\gamma_t\in(0,1]\)，
\[
v_t=(1-\gamma_t)v_{t-1}+\gamma_t q_t,
\quad \delta_t=v_t-\nu_t,\quad \Delta\nu_t=\nu_t-\nu_{t-1}.
\]
则
\[
\boxed{\mathbb E_{t-1}\delta_t^2=
[(1-\gamma_t)(\delta_{t-1}-\Delta\nu_t)+\gamma_t b_t]^2
+\gamma_t^2S_t.}
\tag{10}
\]
并有
\[
\mathbb E_{t-1}\delta_t^2\le(1-\gamma_t)\delta_{t-1}^2+
\frac{2(1-\gamma_t)^2}{\gamma_t}(\Delta\nu_t)^2+
2\gamma_t b_t^2+\gamma_t^2S_t.
\tag{11}
\]

**证明。** 条件期望中只有 \(\zeta_t\) 为新随机量，交叉项为零，得 (10)。对
\((1-\gamma)\delta+\gamma[b-(1-\gamma)\Delta\nu/\gamma]\)
使用平方函数的凸性，再用 \((u-v)^2\le2u^2+2v^2\)，得 (11)。证毕。

若将 v 更新投影至固定区间，且每个 \(\nu_t\) 都在该区间内，(11) 仍成立，因为投影不会增加到区间内点的距离；(10) 的等式一般不再成立。目标在区间外时必须另计裁剪偏差。

若 \(\gamma_t\in[\gamma_-,\gamma_+]\)、\(|\Delta\nu_t|\le D_\nu,|b_t|\le B_\nu,S_t\le S\)，则迭代 (11) 给出
\[
\mathbb E\delta_t^2\le(1-\gamma_-)^t\mathbb E\delta_0^2+
\frac{2D_\nu^2}{\gamma_-^2}+
\frac{2\gamma_+B_\nu^2+\gamma_+^2S}{\gamma_-}.
\tag{12}
\]
固定 \(\gamma\) 的尺度为 \(O(D_\nu^2/\gamma^2+B_\nu^2+\gamma S)\)。长记忆降低随机波动，却放大漂移误差；因此不能由固定点的高 \(\beta_1\) 优势直接推导训练中越长越好。

对原 v2.1，取 \(b_t=2\alpha_t h_te_t\)，\(S_t\) 由 (5) 给出。原偏差修正 v 的有效遗忘率为 \((1-b)/(1-b^t)\)，也能代入。未来自适应权重会依赖过去噪声，不能把随机权重当成确定常数后删去全部跨时项；(10) 避开了这一错误。

## 1.6 定理 E：原算法能正式继承的非凸保证

假设 F 为 L 光滑且下有界，\(\|g_t\|_\infty\le G\) 几乎必然、条件无偏，\(w=0\)。令
\[
Q=\sqrt5G,\ B_g=\sqrt dG,\ R=(1-p)^{-1},\ K=R-1,
\quad d_-=(Q+\varepsilon)^{-1},\ \Delta=F(x_1)-F_*.
\]
对固定步长 \(\eta\)，原算法满足
\[
\begin{aligned}
\frac1T\sum_t\mathbb E\|\nabla F(x_t)\|^2\le{}&
\frac{\Delta}{\eta d_-T}+
\frac{L\eta R^2B_g^2}{d_-\varepsilon^2}(K+R^2/2)\\
&+\frac{RB_g^2}{d_-}(1+LK\eta R/\varepsilon)
\left[\frac{RQ}{\varepsilon^2}(\sqrt{1-b}+2/\sqrt T)+\frac K{\varepsilon T}\right].
\end{aligned}
\tag{13}
\]

**证明要点与完整代入链。** 写 \(D_t=(\sqrt{\widehat v_t}+\varepsilon)^{-1}\)。由 (3)，\(0\le\widehat v_t\le Q^2\)，且
\[
\sum_t\|D_t-D_{t-1}\|_{\rm op}
\le Q\varepsilon^{-2}(T\sqrt{1-b}+2\sqrt T)=:\overline{\mathcal V}_T.
\]
这是由 \(\|\widehat v_t-\widehat v_{t-1}\|_\infty\le Q^2[(1-b)+1/t]\)
和 \(|(\sqrt u+\varepsilon)^{-1}-(\sqrt v+\varepsilon)^{-1}|\le\varepsilon^{-2}\sqrt{|u-v|}\) 累加得到。
设 \(A_t=\eta D_t/(1-p^t)\)、\(A_0=\eta R\varepsilon^{-1}I\)、\(U=\eta R/\varepsilon\)，
\(z_t=x_t-KA_{t-1}m_{t-1}\)。直接代数有
\[
z_{t+1}-z_t=-A_{t-1}g_t+R(A_{t-1}-A_t)m_t,
\quad \|z_t-x_t\|\le KUB_g,
\]
\[
\|z_{t+1}-z_t\|\le R U B_g,\quad
\sum_t\|A_t-A_{t-1}\|\le\eta(R\overline{\mathcal V}_T+K/\varepsilon).
\]
光滑下降、只对可预测 \(A_{t-1}\) 使用无偏性，得到
\[
\eta d_-\sum_t\mathbb E\|\mu_t\|^2\le
\Delta+LTU^2B_g^2(K+R^2/2)
+RB_g^2(1+LKU)\eta(R\overline{\mathcal V}_T+K/\varepsilon).
\]
除以 \(\eta d_-T\) 并代入 U 与变差界即得 (13)。当前分母的依赖项已完整保留，未假设它与梯度独立。证毕。

\(\eta\asymp T^{-1/2},1-b=O(T^{-1})\) 给出 \(O(T^{-1/2})\)。固定 \(b=.999\) 时此界含不消失项，不能以此证明目标的 \(T^{-4/5}\)。高斯固定点结论与本节有界梯度假设不同，必须分别陈述。
