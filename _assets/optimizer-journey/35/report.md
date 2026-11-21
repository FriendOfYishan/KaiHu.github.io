# 共同日程下的一阶/二阶矩消融

全部完成：True。

EE=EMA/EMA；AE=AWA/EMA；EA=EMA/AWA；AA=AWA/AWA。主实验固定共同peak=0.001、2%warmup+cosine、wd=.1、clip1。时间受并行GPU争用影响，不作为效率证据。

```json
{
  "state": "complete",
  "time": 1794498337.4870977
}
```

|方法|完整seed数|最终NLL均值|
|---|---:|---:|
|EE|3|1.556815|
|AE|3|1.614504|
|EA|3|1.553037|
|AA|3|1.615380|

正差表示替换AWA后NLL增加；交互项为AA−AE−EA+EE，正值表示两者同时使用比可加效应更差。
first_with_EMA_second: 0.057689 ± 0.002507
first_with_AWA_second: 0.062342 ± 0.001228
second_with_EMA_first: -0.003778 ± 0.000916
second_with_AWA_first: 0.000875 ± 0.002165
interaction: 0.004653 ± 0.002873

![确认曲线](E:/NewSGD/CrossAdamW/experiments/20261112_moment_ablation/figures/confirmation.png)