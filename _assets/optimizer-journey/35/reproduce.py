"""Original GPU numerical check; requires CUDA and Triton."""
import json
from core import check
print(json.dumps(check(),indent=2))
