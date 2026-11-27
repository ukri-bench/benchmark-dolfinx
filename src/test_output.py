import json
import pprint
import sys

import numpy as np

filename = sys.argv[1]

with open(filename, "r") as f:
    data = json.load(f)

pprint.pprint(data)

input, output = data["input"], data["output"]

# Check size is correct
assert output["ndofs_global"] == 1000

# Check matrix-free and assembled matrix results agree. Compare the
# vectors (relative norm of the difference), not just their norms.
assert input["mat_comp"], "Run with --mat_comp to check correctness"
rtol = 1e-10 if input["scalar_size"] == 64 else 1e-4
assert output["e_norm_rel"] < rtol, f"Relative error {output['e_norm_rel']} >= {rtol}"

# Compare to known result
assert np.isclose(output["y_norm"], 9.912865833415553)
