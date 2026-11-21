"""A mathematical safeguard, not an estimator of unknown neural-network bounds.

Inputs MUST be valid bounds known before the current gradient is sampled.
Plugging in the same noisy current gradient does not preserve the certificate.
See theory/07_DESCENT_GATED_RESEARCH_ZH.md.
"""
import math


def certified_gate(*, alignment_lower, baseline_energy_upper,
                   increment_energy_upper, curvature_upper, lr, max_gate=1.0):
    """Minimize a certified quadratic upper bound on loss versus a reference step.

    Returns (gate, conditional_loss_difference_upper_bound). This does not
    certify an entire trajectory or that the reference step itself descends.
    max_gate can enforce an independently constructed feasible stability interval.
    """
    values=(alignment_lower,baseline_energy_upper,increment_energy_upper,
            curvature_upper,lr,max_gate)
    if not all(math.isfinite(x) for x in values):
        raise ValueError("All certificate inputs must be finite")
    if min(baseline_energy_upper,increment_energy_upper,lr)<0:
        raise ValueError("Energies and learning rate must be nonnegative")
    if curvature_upper<=0 or not 0<=max_gate<=1:
        raise ValueError("Curvature bound must be positive and max_gate in [0,1]")
    if lr==0 or increment_energy_upper==0 or max_gate==0:
        return 0.0,0.0
    certified_linear=alignment_lower-curvature_upper*lr*math.sqrt(
        baseline_energy_upper)*math.sqrt(increment_energy_upper)
    if certified_linear<=0:
        return 0.0,0.0
    gate=min(max_gate,certified_linear/(curvature_upper*lr*increment_energy_upper))
    upper=-lr*gate*certified_linear+0.5*curvature_upper*lr**2*gate**2*increment_energy_upper
    return gate,upper
