# Optimizer-only excerpt; see SOURCE_MANIFEST.json for original SHA-256.
import math

import torch

from torch import nn

from torch.nn import functional as F

B1 = 0.9

RHO = 0.999

EPS2 = 1e-16
