# 原Joint-AWA能量吸收核验

[完整证明与结论](../../theory/26_ENERGY_ABSORPTION_WITHOUT_IDENTIFICATION_ZH.md)。

原算法不变。5000个随机状态验证精确配方，百万步标量扫描核对有效下降系数。Joint从第3步起解析保证theta>=eta/3；渐近theta/eta为3/4，原线性AWA为5/8。它不是20%的训练提速。

2304个旧固定快照中，真实期望势函数下降1779个；直接界证实1216个，吸收界868个，1/3保守界567个，全局曲率界0个。无错误下降证书。吸收界完成理论闭合但比保留全部能量项更松。

![结果](energy_absorption.png)

复现：`python experiments/20261110_energy_absorption/run.py`。这是CPU数学诊断，不是GPU训练或运行速度比较。
