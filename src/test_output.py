import json
import pprint
import numpy as np
import sys

filename = sys.argv[1]

f = open(filename, "r")
data = json.load(f)
f.close()

pprint.pprint(data)

# Check size is correct and matrix-free and CSR agree
assert data["output"]["ndofs_global"] == 1000
assert np.isclose(data["output"]["y_norm"], data["output"]["z_norm"])

# Compare to known result
assert np.isclose(data["output"]["y_norm"], 9.912865833415553)
