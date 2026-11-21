# 34路线的计数校准候选：平稳风险、变化恢复与同阶下界

日期：2026-11-13。

本文保留已有稳定34，不修改训练实现。另提出一个面向在线计数的专用重构：不用二阶矩、调和分母或EMA，以一个计数和生成经过解析校准的参数更新。本文给出自包含的证明及确定性数值核对；没有新增GPU训练，也没有证明神经网络训练优势。

**可成立的主要结论：**在已校准基线、独立Poisson计数、持续向上变化且变化幅度至少为Delta的模型中，可以用常数状态和单次观测，达到平稳均方误差不超过v、恢复延迟随log(Delta²/v)增长。相同风险约束下，固定线性滤波器需要Omega(1/v)的半响应时间；一般因果估计器则受到Omega(log(Delta²/v))的信息下界。下面的构造在固定置信水平、强度范围有界比例时达到后者的阶。

此最优性是指定统计模型及指定风险—恢复准则下的阶最优性；不是原34、所有在线问题、所有噪声模型或所有现有算法上的最优性。

## 1. 为什么只有“单侧界、大梯度有信息”还不够

设平稳观测恒为a。变化后，观测以概率1-q仍为a，以概率q成为a+Delta/q。变化后的均值增加Delta，梯度Y-a非负，而且每次大梯度都是真正的变化证据。然而在d步内看不到任何证据的概率为(1-q)^d。q可以任意小，因此仅由a和Delta无法给出统一的有限恢复保证。

必须明确噪声族、尾部或信息距离。本文选择自然的计数模型，而不是把这个额外条件隐去。

## 2. 问题、梯度与评价标准

观测独立，且

\[
Y_t\sim\operatorname{Pois}(\lambda_t),\qquad
\lambda_t=\begin{cases}a,&t<\nu,\\ \lambda_1,&t\ge\nu,\end{cases}
\]

其中a>0是已校准基线，nu未知，变化后强度未知但满足

\[
0<\Delta\le a,\qquad a+\Delta\le\lambda_1\le\Lambda.
\]

Delta是需要保证恢复的最小变化幅度，不是事先知道的实际跳变量。Lambda是强度上界。实际应用可包括请求到达率、计数传感器、故障或事件频率；Poisson拟合及独立性需要验证。

负对数强度x的单次损失与梯度为

\[
\ell(x;Y)=e^{-x}+Yx,\qquad g=Y-e^{-x}.
\]

若预测强度限制在[a/2,2Lambda]，自然有g>=-2Lambda。本文直接更新预测强度theta，再按需令x=-log(theta)，不再对x做Adam式欧拉更新。已经获得g与当前theta时，Y=g+theta可直接恢复，不需要第二次梯度计算。

### 2.1 平稳风险

记P0为始终lambda=a的分布，要求每个确定时刻t满足

\[
\mathbb E_0(\widehat\lambda_t-a)^2\le v.
\]

该要求同时控制方差与均值偏差。只要求“无偏”不足以表达稳定性。

### 2.2 恢复

在任意变化时刻nu之后，要求至多d步后，对于每个确定的后续时刻，

\[
\Pr_{\nu,\lambda_1}(|\widehat\lambda_t-\lambda_1|>\Delta/4)\le\alpha,
\qquad t\ge\nu+d-1.
\]

这里是逐时刻的概率保证，不是无限时间内永不再次越界的保证。若要求同时覆盖有限H个后续输出，用alpha/H替换alpha即可，代价包含log H。

## 3. 原34和未经校准的快速响应的缺口

原稳定34的归一化形式是

\[
u=\frac{\mu}{\sqrt{\sum_k a_k\phi_c(g_k)}+\epsilon},\qquad
\phi_c(g)=\frac{cg^2}{c+g}.
\]

其正向大梯度尺度注入是近似线性而不是平方，但随机分母与分子共用观测会产生校准偏差。例如epsilon=0、首步g=+h或-h各半、0<h<c，有

\[
\mathbb E u=\tfrac12[\sqrt{1+h/c}-\sqrt{1-h/c}]>0.
\]

这是原34的一个明确障碍，不是完整轨迹发散定理。

另一个看似合理的更新是“计数足够大才相信当前均值”。设长度b的块内总计数S~Pois(b lambda)，响应函数psi取值[0,1]并单调增加，直接写

\[
U_{\rm naive}(S)=a+(S/b-a)\psi(S).
\]

在lambda=a时，它的偏差精确为

\[
\mathbb E_a U_{\rm naive}-a
=a\,\mathbb E_a[\psi(S+1)-\psi(S)]\ge0.
\]

所以“仅更换一个数据依赖权重”并没有解决问题。下一节减掉的是这项选择偏差的逐样本校正量。

## 4. 候选的完整形式

选固定块长b。阈值与过渡宽度预先计算为

\[
k=\left\lceil b(a+\Delta/2)\right\rceil,
\qquad L=\max\{2,\lceil b\Delta/8\rceil\},
\]

\[
\psi(s)=\left[\frac{s-k}{L}\right]_0^1,
\qquad[z]_0^1=\min\{1,\max\{0,z\}\}.
\]

第j个不重叠块的计数和为S_j。核心更新为

\[
\boxed{
U_j=a+\frac{S_j}{b}\psi(S_j-1)-a\psi(S_j).
}
\tag{A}
\]

实际有界预测采用

\[
\boxed{
\widehat\lambda_{jb}=\min\{2\Lambda,U_j\},
\qquad x_{jb+1}=-\log\widehat\lambda_{jb}.
}
\tag{B}
\]

在两次块输出之间保持最近预测；第一个完整块前输出a。S仅通过S<-S+Y累加，输出后归零。x的log只在需要该参数化时于块末计算；直接用强度预测时无需log。

(A)也可写为

\[
U_j=a+(S_j/b-a)\psi(S_j)
-\underbrace{\frac{S_j}{b}[\psi(S_j)-\psi(S_j-1)]}_{\text{同一观测的解析偏差校正}}.
\]

psi(S-1)只是把已有整数统计量减1后代入一个分段线性函数，既不是重新采样，也不是在新参数点求梯度。

这个构造继承的是34“有信息的正向变化不应被平方分母长期压制”的目标，**并非原34的代数等价化简**。它改变了更新对象与随机校准机制。

### 4.1 为什么是缓坡而不是硬阈值

硬阈值的校正可能在边界处给出0强度。宽度L>=2的缓坡保持正值，并限制相邻计数导致的响应跳变。任何从0到1、在L个整数步内完成的函数，都满足

\[
\max_s|\psi(s+1)-\psi(s)|\ge1/L.
\]

线性缓坡达到等号。因此它在这个明确的最大相邻跳变准则下最优；不据此声称它最小化了所有Poisson加权风险。

### 4.2 正值性

当S<=k时U=a；当S>=k+L+1时U=S/b。
中间令S=k+j，1<=j<=L，则

\[
U=a+\frac{(k+j-ba)j-(k+j)}{bL}.
\]

分子的相邻增量为2j+k-ba>=0，第一个值给出U=a(1-1/L)。故

\[
\boxed{a/2\le U\le\max\{a,S/b\}.}
\]

截顶后预测属于[a/2,2Lambda]，因此梯度确有自然的固定单侧界。

### 4.3 为何阈值附近会有小幅向下校正

这项更新不是处处向上。在第一个响应计数S=k+1处，U=a-a/L；最大的基线下回调幅度不超过a/L。这是显式可控的代价。

有一个简单的不可能性结果：若某估计器在所有计数上都满足h(S)>=a，同时要求E_a h(S)=a，则h(S)=a在P_a下几乎处处成立。由于Poisson(a b)在每个非负整数上都给正概率，h实际上只能恒等于a，无法对任何上升计数响应。因此非平凡、基线严格无偏的更新必须允许某些样本上的向下补偿。将本文输出再截成max(a,U)，会破坏严格校准。

## 5. 两个精确恒等式

### 定理1：均值响应恒等式

对任意满足可积性的psi、任意lambda>0，令S~Pois(b lambda)，则

\[
\boxed{\mathbb E_\lambda U
=a+(\lambda-a)\mathbb E_\lambda\psi(S).}
\tag{C}
\]

特别地，未截顶核心在基线严格无偏：E_a U=a。

证明：Poisson求和移位给出

\[
\mathbb E[S\psi(S-1)]=b\lambda\mathbb E\psi(S).
\]

代入(A)即可。这个基础恒等式是已有的Poisson/Hudson恒等式，不能作为新发现；例如[Yuan (2005), Eq. (11)](https://pages.stat.wisc.edu/~wahba/stat860public/pdf1/yuan.poisson.pdf)给出了相同的求和关系。本文使用它构造参数更新，而不是估计二阶矩。

### 定理2：基线均方误差精确分解

在lambda=a、mu=ba时，

\[
\boxed{
\mathbb E_a(U-a)^2
=\frac ab\mathbb E_a\psi(S)^2
+a^2\mathbb E_a\!\left[(\psi(S+1)-\psi(S))^2\right].
}
\tag{D}
\]

第一项是估计噪声，第二项是数据依赖响应的变化代价。这是本构造的核心可检验等式，不能把第二项省去。

证明：令Z=S psi(S-1)-mu psi(S)。有

\[
\begin{aligned}
\mathbb E S^2\psi(S-1)^2
&=\mu^2\mathbb E\psi(S+1)^2+\mu\mathbb E\psi(S)^2,\\
\mathbb E S\psi(S-1)\psi(S)
&=\mu\mathbb E\psi(S)\psi(S+1).
\end{aligned}
\]

展开E Z²并除以b²，即得(D)。

顺带地，若psi从0到1的差分仅落在s=k,...,k+L-1，且p_s=Pr_a(S=s)，则加权Cauchy不等式给出

\[
\sum_{s=k}^{k+L-1}p_s[\psi(s+1)-\psi(s)]^2
\ge\left(\sum_{s=k}^{k+L-1}p_s^{-1}\right)^{-1}.
\]

这说明校准并非免费：有响应变化就有正的风险代价。只最小化这个差分项时，增量正比于1/p_s；它不必最小化(D)的完整风险，也不如线性缓坡便宜。

## 6. 平稳风险、偏差与恢复上界

采用下列易核查的Poisson尾界：若S~Pois(b lambda)，z>0，则

\[
\Pr(S/b-\lambda\ge z)\le
\exp\{-bz^2/[2(\lambda+z)]\},
\]

\[
\Pr(S/b-\lambda\le-z)\le
\exp\{-bz^2/(2\lambda)\}\quad(0<z\le\lambda).
\]

可由矩母函数及Chernoff优化直接推出；这里使用宽松常数以简化证明。

### 6.1 平稳风险

psi(S)和其向前差分仅在S>=k时可能非零，差分不超过1/L。因此(D)推出

\[
R_a(b):=\mathbb E_a(U-a)^2
\le\left(\frac ab+\frac{a^2}{L^2}\right)\Pr_a(S\ge k)
\]

\[
\boxed{
R_a(b)\le
\left(\frac ab+\frac{64a^2}{b^2\Delta^2}\right)
\exp\left\{-\frac{b\Delta^2}{12a}\right\}.
}
\tag{E}
\]

其中用到了Delta<=a及k>=b(a+Delta/2)。若b>=64a/Delta²，则

\[
R_a(b)\le\frac{2a}{b}\exp\{-b\Delta^2/(12a)\}.
\]

由于a<=Lambda<2Lambda，截顶只会降低在真值a处的平方误差，所以(B)满足同样的风险上界。

### 6.2 截顶引入的可控偏差

核心U在基线严格无偏；实际有界预测的偏差为

\[
\mathbb E_a\widehat\lambda-a=-\mathbb E_a(U-2\Lambda)_+.
\]

由U<=max(a,S/b)及Cauchy不等式，

\[
0\le a-\mathbb E_a\widehat\lambda
\le\sqrt{a/b}\exp\{-b I(2\Lambda\Vert a)/2\},
\]

其中I(u||a)=u log(u/a)-u+a。因此有界版本的偏差不是严格零，但随b指数衰减。不能在应用截顶之后仍宣称精确无偏。

### 6.3 变化后恢复

若b>=64a/Delta²，则b Delta>=64，取整不会破坏

\[
k+L+1\le b(a+3\Delta/4).
\]

在一个完全位于变化后的块内，若|S/b-lambda_1|<=Delta/4，则

\[
S\ge b(\lambda_1-\Delta/4)\ge b(a+3\Delta/4)\ge k+L+1.
\]

故psi(S)=psi(S-1)=1，核心U=S/b。截顶不会增加相对lambda_1<=Lambda的误差。于是

\[
\boxed{
\Pr(|\widehat\lambda-\lambda_1|>\Delta/4)
\le2\exp\{-b\Delta^2/(40\Lambda)\}.
}
\tag{F}
\]

任意变化时刻之后，至多2b步即可完成第一个纯变化后块；以后每个完整块也满足(F)。此前包含变化点的混合块不用纳入证明。

变化后的核心偏差也可直接控制。由(C)，

\[
0\le\lambda_1-\mathbb E_{\lambda_1}U
\le(\lambda_1-a)\exp\{-b\Delta^2/(40\Lambda)\}.
\]

截顶的额外偏差至多

\[
\sqrt{\Lambda/b}\exp\{-b\Lambda/8\}.
\]

这里让预测上限是2Lambda而不是真值上限Lambda，保留了裕量，避免真值恰在上限时截顶造成仅按1/sqrt(b)衰减的边界偏差。

### 定理3：一个显式的风险—恢复选择规则

设0<alpha<1/4，v>0。选择

\[
\boxed{
b=\left\lceil\max\left\{
\frac{64a}{\Delta^2},\;
\frac{12a}{\Delta^2}\log_+\frac{\Delta^2}{v},\;
\frac{40\Lambda}{\Delta^2}\log\frac2\alpha
\right\}\right\rceil,
}
\tag{G}
\]

其中log_+(z)=max(0,log z)。则对所有确定时刻，平稳风险不超过v；对所有允许的变化时刻及变化后强度，恢复延迟不超过2b，成功概率至少1-alpha。

证明：b>=64a/Delta²使2a/b<=Delta²/32；若v<Delta²，指数项至多v/Delta²，故(E)<=v/32<=v；若v>=Delta²，则(E)<=Delta²/32<=v。恢复由(F)及(G)第三项得到。

因此

\[
d=O\left(1+\frac a{\Delta^2}\left[1+\log_+\frac{\Delta^2}{v}\right]
+\frac\Lambda{\Delta^2}\log\frac1\alpha\right).
\tag{H}
\]

### 6.4 与真实预测损失连接

对lambda在[a,Lambda]、预测theta在[a/2,2Lambda]，Poisson总体损失差为

\[
F_\lambda(-\log\theta)-F_\lambda(-\log\lambda)
=\theta-\lambda+\lambda\log(\lambda/\theta)
\le\frac{(\theta-\lambda)^2}{\theta}
\le\frac2a(\theta-\lambda)^2.
\]

故平稳期的期望超额损失不超过2v/a；变化恢复事件上超额损失不超过Delta²/(8a)。这使“恢复”与任务损失挂钩，但并非一般非凸优化收敛率。

## 7. 固定线性记忆的局限

考虑固定因果线性估计器

\[
\widehat\lambda_t=a+\sum_{j\ge0}w_j(Y_{t-j}-a),
\]

其中权重不依赖观测或变化时刻，平方可求和。普通EMA、固定窗口均值属于此类；证明不依赖非负性，但讨论平均器时通常w_j>=0、sum w_j=1。

平稳风险为a sum w_j²。若不超过v，变化发生d步后，均值响应比例为W_d=sum_{j=0}^{d-1}w_j。Cauchy给出

\[
W_d^2\le d\sum_j w_j^2\le dv/a.
\]

**定理4：**若要完成至少一半均值响应，即W_d>=1/2，则

\[
\boxed{d\ge a/(4v).}
\tag{I}
\]

这在允许固定线性估计器同样知道a的情况下依然成立，并不是给候选额外基线信息、不给对照。

还可得到与第2节同口径的概率下界。若d<=a/(16v)，则W_d<=1/4。取lambda_1=a+Delta，变化后的估计均值至多a+Delta/4，方差至多Lambda v/a。因此

\[
\Pr(|\widehat\lambda_t-(a+\Delta)|\le\Delta/4)
\le\frac{4\Lambda v}{a\Delta^2}.
\]

当右端小于1-alpha时，固定线性估计器不可能在这么短时间内达到恢复要求，即同口径恢复也需要Omega(a/v)。

EMA的显式例子：w_j=(1-beta)beta^j，风险a(1-beta)/(1+beta)，半响应延迟ceil(log 2/|log beta|)，与(I)一致。

**比较范围：**这是固定线性记忆的下界，不是完整AdamW参数轨迹、CUSUM、任意自适应重启算法或所有现代优化器的下界。

## 8. 任何因果算法都绕不过的信息下界

设任意因果估计器可使用历史、已知a、随机化和任意计算，平稳风险仍<=v。比较P0（始终a）和P1（从nu起a+Delta）。在第d个变化后观测时，考虑事件

\[
A=\{\widehat\lambda\ge a+\Delta/2\}.
\]

由平稳风险及Markov不等式，P0(A)<=4v/Delta²。若恢复概率>=1-alpha，则P1(A)>=1-alpha。

d个变化后观测的KL散度恰为

\[
dI,\qquad I=(a+\Delta)\log(1+\Delta/a)-\Delta.
\]

数据处理到二元事件A，再用二元熵至多log 2，得到

\[
\boxed{
d\ge\frac{(1-\alpha)\log[\Delta^2/(4v)]-\log2}{I}.
}
\tag{J}
\]

当右端为负时只保留d>=0的平凡结论。证明不要求估计器是线性或采用某种门控。

积分二阶导数还给出

\[
\frac{\Delta^2}{2(a+\Delta)}\le I\le\frac{\Delta^2}{2a}.
\]

因此在alpha固定、Lambda/a有界、v/Delta²趋于0的非平凡区域，结合(H)与(J)，该问题的最优阶为

\[
\boxed{
d^*(v,\Delta)
=\Theta\left(\frac a{\Delta^2}\log\frac{\Delta^2}{v}\right),
}
\tag{K}
\]

若需要覆盖参数同时变化导致的不足一步情形，应在量级表达式中保留max(1,...)及取整。这里没有声称最优常数，也没有匹配alpha趋于0时的完整多参数下界。

这是“达到一般信息下界的阶、突破固定线性记忆的界”，而不是“突破信息下界”。已有最快变点检测也具有对数型延迟—误报权衡，例如[Wu et al. (2023), Section 2](https://proceedings.mlr.press/v206/wu23b/wu23b.pdf)回顾了CUSUM及其经典最优性。不能把对数阶本身描述为首次发现。

## 9. 已知基线是重要条件，不可掩盖

候选对参考值a严格校准，不会在所有未知lambda处同时严格无偏。事实上，对一个Poisson块统计量S，如果估计器h(S)对每个lambda>0都满足E_lambda h(S)=lambda，则Poisson族的完备性（或比较幂级数系数）推出h(S)=S/b。因此不能既要求整个Poisson参数族上严格无偏，又在基线处任意压低风险并保留快响应。

实际只能由独立校准期得到参考a时，(C)仍然有效：若当前真实强度为lambda，

\[
\mathbb E_\lambda U-\lambda=(a-\lambda)[1-\mathbb E_\lambda\psi(S)].
\]

所以核心的均值偏差绝对值不超过参考误差|a-lambda|。但这不代表原风险v与原恢复常数原封不动成立；阈值余量、基线估计误差及校准样本成本必须计入。基线来自同一个当前块时，本文的条件期望证明也不再直接适用。

下降变化、小于Delta的缓慢漂移、多个相邻未知基线、过度离散计数、依赖计数以及带未知特征的Poisson回归，不在(K)的定理范围内。把a自适应替换成上一预测也需要重新证明，不能直接套用本结果。

## 10. 计算与空间

每个观测只更新一个计数和；另有块计数器和当前预测。块末评价两次相邻的分段线性函数并做常数次乘加、截顶。1/b、1/L及阈值可提前计算。

- 单序列每步O(1)算术、O(1)状态；核心没有逐步平方、开方、随机数、第二梯度或二阶矩状态。
- d个独立计数通道为O(d)工作量和O(d)状态，不能把维数忽略后声称任意高维问题O(1)。
- 与稳定34相比减少了逐坐标第二统计状态；模型参数/当前预测以及原始数据成本仍存在。
- 大多数步骤只做累加，代价是预测按块更新。恢复上界中的2b已经计入这项等待，不是把批处理延迟从指标中删去。
- 这只说明算术及状态结构更省；尚未计时，不报告GPU或端到端加速倍数。

## 11. 确定性数值核对

运行`python theory/verify_34_count_calibration.py`，直接对Poisson概率质量求和，没有模拟抽样误差。结果保存在[核对记录](34_COUNT_CALIBRATION_CHECKS.json)。

36组参数核对了(C)、(D)、未经校准更新的偏差公式、正值性、截顶不增加MSE和(E)。均值恒等式的最大绝对数值误差约2.17e-11。

例：a=4，Delta=2，Lambda=6，alpha=.05，v=1e-5。

|量|结果|含义|
|---|---:|---|
|(G)给出的b|222|保守、统一于允许的lambda_1的选取|
|候选恢复上界|444步|2b，达到绝对误差<=.5的概率>=95%|
|固定线性方法半均值响应下界|100000步|更弱的恢复目标仍受此约束|
|一般因果方法信息下界|约23.67步，即整数至少24步|不是宣称候选只需24步|
|b=222时，基线实际MSE数值和|约2.17e-15|上界常数较保守|
|b=222、lambda_1=6时恢复概率数值和|约0.997744|单个完整变化后块|

脚本另外扫描了同一模型中单个变化后强度lambda_1=6的可行块长，仅用于核查界的保守程度。它不替代对所有lambda_1的定理，也不是未知任务上的训练结果。

## 12. 贡献应如何表述，下一步如何证伪

当前可以作为一项明确数学结果来讨论的是：

1. 固定线性记忆的风险—恢复障碍，以及不受此类限制的一般信息下界。
2. 一项无需新样本/梯度的解析校正，把大计数响应与平稳偏差分离。
3. 校正后的精确风险恒等式，把响应敏感度的代价显式纳入界。
4. 一个正值、有界、常数状态的实现，在指定的分离Poisson模型中达到信息下界的阶。

其中Poisson恒等式、变点检测思想、KL两点下界方法都属于已有工具。本文推导了这套具体更新与评价准则的结果，尚不能据此认定该组合在全部文献中首次出现，或据此声称胜过已有最优变点方法。

下一轮最有信息量的比较应包括：原稳定34、调优EMA、未校准缓坡、本文校正缓坡，以及CUSUM/重启均值。先在相同校准信息、相同平稳MSE预算下比较恢复延迟；再逐一破坏Poisson、独立性、基线精度和最小跳变假设。去掉校正项的消融应首先检查偏差公式，而不能只看平均恢复速度。

如果仅赢EMA而不能在校准、常数、状态开销或失配稳健性上优于成熟变点方法，则论文贡献应停留在这里，不能把已有对数最优阶当作新的普适边界。
