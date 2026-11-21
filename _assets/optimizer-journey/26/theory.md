# 不识别漂移、不改递推：Joint-AWA 的能量吸收与闭合下降界

2026-11-10。承接第23–25版。原Joint-AWA保持不变。本稿解决第23版中的预条件器能量变化项V如何统一控制的问题，不增加梯度、向量状态或在线标量控制器。文献新颖性未认定，未证明优于AdamW。

## 1. 本次推进

此前尝试用创新相关性识别历史偏差，但低成本投影没有通过跨任务门槛。本稿不再要求识别漂移：利用原二阶矩递推对预条件器增长的限制，将V与原有动量耗散配方。

得到三个结果：

1. 一个精确能量配方，将未受控的V替换成两个非负残差与确定性有效下降系数theta_t。
2. 对当前Joint日程，从t=3起严格有theta_t>=eta_t/3，不需要估计真实梯度或噪声。
3. 同一证明中的系数：Joint的theta_t/eta_t趋于3/4，原线性AWA趋于5/8。这是能量分析系数的区别，不是训练速度、复杂度或泛化的比例提升。

在有界梯度等附加条件下得到原候选的闭合非凸下降与O(T^(-1/3))加权平稳性保证。该主导阶数没有超过第18版；改进在于能量项闭合和分析不依赖在线偏差识别。

## 2. 算法、记号与起点

原候选：
\[
\alpha_t=\max\{2/(t+1),0.1t^{-1/3}\},\quad
\eta_t=\eta_0t^{-1/2},
\]
\[
m_t=(1-\alpha_t)m_{t-1}+\alpha_tg_t,\quad
v_t=(1-\alpha_t)v_{t-1}+\alpha_tg_t^2,
\quad x_{t+1}=x_t-\eta_tD_tm_t,
\]
\[
D_t=\operatorname{diag}((\sqrt{v_t}+\varepsilon)^{-1}),\quad
m_0=v_0=0,\quad\varepsilon>0.
\]
F为L光滑，无权重衰减，g_t=grad F(x_t)+xi_t，条件零均值噪声。需要收敛结论时再增加下有界和梯度有界假设。

沿用第23版：
\[
q_t=\frac{\eta_t(1-\alpha_t)}{2\alpha_t},\quad
\Psi_t=F(x_t)+q_{t-1}\|m_{t-1}\|_{D_{t-1}}^2,
\]
\[
M_t=\|m_t\|_{D_t}^2,\quad
I_t=\|m_t-m_{t-1}\|_{D_t}^2,\quad
U_t=\|D_tm_t\|^2,
\]
\[
V_t=m_{t-1}^T(q_tD_t-q_{t-1}D_{t-1})m_{t-1}.
\]
已知逐路径下降界：
\[
\Psi_{t+1}-\Psi_t
\le-\eta_tM_t-q_tI_t+V_t+\frac L2\eta_t^2U_t
+\eta_t\langle\xi_t,D_tm_t\rangle.\tag{2.1}
\]

## 3. 二阶矩给出了预条件器的增长上限

令beta_t=1-alpha_t。因为v_t>=beta_t v_{t-1}逐坐标成立，且epsilon>0，
\[
\sqrt{\beta_t}D_t
\preceq\operatorname{diag}\frac1{\sqrt{v_{t-1}}+\varepsilon/\sqrt{\beta_t}}
\preceq D_{t-1}.
\]
因此对t>=2，
\[
\boxed{D_{t-1}\succeq\sqrt{1-\alpha_t}\,D_t.}\tag{3.1}
\]
这个关系并不要求D_t单调。即使v下降、D上升，上升幅度仍受原递推约束。

## 4. 精确配方：将V吸收进已有动量耗散

对t>=3，q_{t-1}>0。定义纯日程量
\[
k_t=q_{t-1}\sqrt{1-\alpha_t}>0,\qquad
\boxed{\theta_t=\eta_t+q_t-\frac{q_t^2}{k_t}.}\tag{4.1}
\]
再定义两个仅用于证明的非负量
\[
\mathcal R_t=k_t\left\|\frac{q_t}{k_t}m_t-m_{t-1}\right\|_{D_t}^2,
\]
\[
\mathcal W_t=q_{t-1}m_{t-1}^T
\left(D_{t-1}-\sqrt{1-\alpha_t}D_t\right)m_{t-1}.
\]
R>=0由平方范数，W>=0由(3.1)。直接展开得到精确恒等式
\[
\boxed{-\eta_tM_t-q_tI_t+V_t
=-\theta_tM_t-\mathcal R_t-\mathcal W_t.}\tag{4.2}
\]

证明：令m=m_t、b=m_{t-1}、D=D_t。展开左侧，m^TDm的系数为-(eta+q)，交叉项为2q m^TDb，旧动量项为-q_{t-1}b^TD_{t-1}b。右侧展开后系数完全一致，因为theta+q^2/k=eta+q且k=q_{t-1}sqrt(1-alpha)。

代入(2.1)：
\[
\boxed{\Psi_{t+1}-\Psi_t
\le-\theta_tM_t-\mathcal R_t-\mathcal W_t
+\frac L2\eta_t^2U_t+\eta_t\langle\xi_t,u_t\rangle.}\tag{4.3}
\]
因此V不需要单独识别，也不需要假装V<=0。R、W不要求优化器实际计算。

## 5. 当前Joint日程严格保留正的有效下降系数

### 定理1

对所有整数t>=3，当前Joint日程满足
\[
\boxed{\frac{\theta_t}{\eta_t}\ge\kappa_t^{\rm linear}\ge\frac13,}\tag{5.1}
\]
其中
\[
\kappa_t^{\rm linear}
=\frac{t+3}{4}-\frac{(t-1)^2}{4(t-2)}\sqrt{1+\frac1t}.
\]

### 证明

记A_t=(1-alpha_t)/(2alpha_t)，则
\[
A_t=\min\{(t-1)/4,\;5t^{1/3}-1/2\}.
\]
函数(5t^(1/3)-1/2)/(t-1)在t>1递减：其导数的分子为
\(-\frac{10}{3}t^{1/3}-\frac53t^{-2/3}+\frac12<0\)。于是
\[
A_t\le(t-1)/4,\qquad
\frac{A_t}{A_{t-1}}\le\frac{t-1}{t-2}.
\]
又eta_t/eta_{t-1}=sqrt((t-1)/t)，而sqrt(1-alpha_t)=sqrt(2A_t/(1+2A_t))。代入theta定义，得到
\[
\frac{\theta_t}{\eta_t}
=1+A_t-\frac{A_t}{A_{t-1}}\sqrt{\frac{t-1}{t}}\sqrt{A_t^2+A_t/2}
\ge 1+A_t-C_t\sqrt{A_t^2+A_t/2},
\]
其中C_t=((t-1)/(t-2))sqrt((t-1)/t)>1。函数f(A)=1+A-C_t sqrt(A^2+A/2)在A>0严格递减，因此取A_t上界(t-1)/4得到kappa_linear。

最后用sqrt(1+1/t)<=1+1/(2t)：
\[
\kappa_t^{\rm linear}\ge\frac58-\frac{2t+1}{8t(t-2)}\ge\frac13\quad(t\ge3).
\]
最后一步等价于7t^2-20t-3>=0，在t=3取等号且之后递增。证毕。

这是一份全时域证明，后面的百万步扫描只是核验，不是证明的替代品。t=1、2为有限启动前缀，本定理不覆盖它们；不能把从t=3成立写成每一步都成立。

## 6. 与原线性AWA的明确区别

原线性AWA的alpha_t=2/(t+1)、eta_t=eta_0/sqrt(t)给出上节的精确kappa_linear，因此
\[
\lim_{t\to\infty}\frac{\theta_t^{\rm linear}}{\eta_t}=\frac58.
\]

Joint后期alpha_t=c t^(-p)，p=1/3<1。此时q_t≈eta_t/(2alpha_t)，
\(q_t/q_{t-1}=1+O(1/t)\)，\(1/\sqrt{1-\alpha_t}=1+\alpha_t/2+O(\alpha_t^2)\)。从而
\[
\frac{\theta_t}{\eta_t}
=1-\frac{q_t}{\eta_t}\left(\frac{q_t/q_{t-1}}{\sqrt{1-\alpha_t}}-1\right)
\longrightarrow1-\frac14=\frac34,
\]
因为1/(t alpha_t)->0。

\[
\boxed{\text{Joint：}\ 3/4\qquad\text{原线性AWA：}\ 5/8.}\tag{6.1}
\]
意义：在同一种能量配方里，Joint较慢的记忆增长使时间系数变化的代价渐近消失，留下更大的有效下降系数。

不能将(3/4)/(5/8)=1.2解释成训练快20%。两算法的M、U、噪声相关项和轨迹不同，本文没有给出目标值排序或迭代复杂度优势。

## 7. 闭合的随机下降界

第23版给出
\[
|E_{t-1}\langle\xi_t,u_t\rangle|
\le\sum_i H_{t,i}\nu_{t,i}.
\]
利用方向有界，可将每项替换成min(H_i nu_i,sqrt(nu_i))以改善有限状态诊断。因此(4.3)给出
\[
\boxed{E_{t-1}\Delta\Psi_t
\le-\theta_t E_{t-1}M_t+\frac L2\eta_t^2E_{t-1}U_t
+\eta_t\sum_i\min(H_{t,i}\nu_{t,i},\sqrt{\nu_{t,i}}).}\tag{7.1}
\]
本式丢掉非负R、W，所以比保留它们的第23版直接界更松；价值是消除了V的未知符号和未闭合控制。更容易证明不等于数值更准确。

### 不需要在线知道噪声方差的全程理论

再假设F下有界、|g_{t,i}|<=G、E_{t-1}||xi_t||^2<=sigma^2。梯度有界只用于理论，不加入裁剪操作。

对当前梯度支持集[-G,G]，同权两矩保证归一化函数的分子绝对值不超过根号项。由导数原式可得
\[
|\phi_i'(z)|\le\alpha_t\left(\frac1\varepsilon+\frac G{\varepsilon^2}\right)
=\alpha_t C_\varepsilon.
\]
故相关项<=C_epsilon alpha_t sigma^2。又U_t<=M_t/epsilon。若
\[
\eta_t\le\frac{\varepsilon}{3L},
\]
则由theta>=eta/3，
\[
\boxed{E\Delta\Psi_t\le-\frac{\eta_t}{6}EM_t
+C_\varepsilon\sigma^2\eta_t\alpha_t.}\tag{7.2}
\]
取t_0>=3，使上述步长条件自t_0起成立。由Psi>=F_*，求和得
\[
\sum_{t=t_0}^T\eta_tEM_t
\le6\{E\Psi_{t_0}-F_*+C_\varepsilon\sigma^2\sum_{t=t_0}^T\eta_t\alpha_t\}.
\]
这里sigma、L只用于分析，算法不读取或估计它们。由于eta alpha≈t^(-5/6)，sum eta≈sqrt(T)，加权M的平均为O(T^(-1/3))。

此外D_t>=1/(G+epsilon)I；结合第18版原算法跟踪误差E||m-grad F||^2=O(t^(-1/3))，得到
\[
\boxed{\frac{\sum_{t=t_0}^T\eta_tE\|\nabla F(x_t)\|^2}{\sum_{t=t_0}^T\eta_t}
=O(T^{-1/3}).}\tag{7.3}
\]
证明最后一步使用||grad F||^2<=2||m||^2+2||m-grad F||^2，及||m||^2<=(G+epsilon)M。

该结论没有比第18版更快，也不是O(1/T)。特别是epsilon很小时，t_0可能极大、C_epsilon很大；不能将这个理论条件直接用作实用学习率推荐。

## 8. 为什么这也没有证明优于AdamW

能量配方本身并非只适用于同权两矩。若一阶新权重为a、二阶新权重为b，预条件器关系改为D_prev>=sqrt(1-b)D；q仍由a决定。固定系数、缓慢变化步长的渐近有效系数为
\[
1+A-\frac A{\sqrt{1-b}},\qquad A=\frac{1-a}{2a}.
\]
其为正的条件是b<4a/(1+a)^2。对常见a=.1、b=.001，这一忽略启动修正的模型系数约.998，反而高于Joint的.75。

这只比较能量系数，不比较两者全程效果；但足以阻止把本定理误读为“Joint比AdamW稳定或更快”。同权重的价值还包括Jensen方向有界和参数简化，不能单凭某一个证书系数给优化器排名。

## 9. 已运行核验

代码与数据：`experiments/20261110_energy_absorption/`。

### 代数与日程

- 5000个7维随机合法状态，检查(4.2)、R/W非负及theta>=eta/3。最大归一化恒等式残差约6.54e-15。
- 扫描t=3至1,000,000，最小theta/eta出现在t=3，约0.345299，大于1/3；Joint系数始终不低于原线性AWA系数。
- t=1024：Joint约.758625，原线性约.624786；t=1,000,000分别约.750146、.625000。

### 同一固定状态、同一势函数的证书

复用两轮旧快照，排除t=1后剩2304个状态。每个状态精确枚举4种本步梯度，实现不改变算法。其中1779个状态真实期望势函数下降，1704个状态V>0。

| 证书 | 证实势函数下降数 | 错误下降证书 |
|---|---:|---:|
| 第23版直接界，保留能量项 | 1216 | 0 |
| 吸收V后的精确theta界 | 868 | 0 |
| 将theta进一步替换成eta/3 | 567 | 0 |
| 再使用全局曲率U<=M/epsilon | 0 | 0 |

这里所有方法使用同一个有界方向噪声修正，比较同一个Psi，不将目标下降和势函数下降混为一谈。

512个无噪声状态中，V全部为正，但吸收界仍证实414个下降。这直接说明“V为正就无法分析”是不成立的；需要联合看动量耗散。

全局安全步长条件在这些快照上一次也未满足，解释了最后一行的失败。该条件足以证明渐近收敛，却不能预测当前有限步训练。

![能量吸收](../experiments/20261110_energy_absorption/energy_absorption.png)

## 10. 当前决定

保留原Joint-AWA，不新增识别器、外推或投影。第23版曾留下的V项现在已通过原递推自身的结构闭合。

当前新增成果是：无识别器的能量吸收恒等式、全时域正系数证明、与原线性AWA的能量系数差异，以及附加假设下的完整下降/收敛论证。尚未获得新优化器、优于AdamW的证明或可用的全局学习率上界。

若继续研究，应优先缩小曲率常数的保守性，同时保留这个已证明的结构，而不是再增加一个未经验证的漂移识别模块。
