"""New small pre-update tracking diagnostic on a quadratic, not the historical training table."""
import numpy as np
rng=np.random.default_rng(19);x=np.ones(4);m=np.zeros(4)
for t in range(1,101):
    true_gradient=x.copy();g=true_gradient+rng.normal(size=4)*.1
    m=.9*m+.1*g
    before=float(np.sum((m-true_gradient)**2))
    x-=.01*m
    after=float(np.sum((m-x)**2))
print({"tracking_at_sample_point":before,"different_metric_after_move":after})
