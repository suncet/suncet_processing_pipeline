# SunCET colormap

In this repository, **"SunCET colormap" means Golden Hour** (voting palette
`sunset-gold`, option 13). Use `suncet_colormap.csv` in this directory as the
canonical, reusable RGB lookup table, not a table regenerated from a screenshot.
An identical copy is maintained in the simulator and processing-pipeline repos.

The CSV has a `red,green,blue` header and 256 rows of integer sRGB values in
0-255, ordered from lowest to highest intensity. Its SHA-256 is
`b71410b6f96ced832173e27776ad160d444d7ffa67d6444f4c64e5877000ee94`.

```python
from pathlib import Path
import numpy as np
from matplotlib.colors import ListedColormap

# Relative to the repository root; anchor to your code's repo path as needed.
path = Path("docs/assets/colormaps/suncet_colormap.csv")
suncet_colormap = ListedColormap(
    np.loadtxt(path, delimiter=",", skiprows=1) / 255.0,
    name="suncet",
)
```

The colormap specifies colors only. Choose the intensity normalization, limits,
orientation, and any spatial filtering separately. Asking to "use the SunCET
colormap" does not imply changing any of those settings.

Original evenly spaced interpolation anchors, low to high:
`#05030a`, `#26102c`, `#65244e`, `#b6444b`, `#ee7950`, `#ffbd69`, `#fff2c9`.
The CSV preserves the exact exported 8-bit table used in the voting gallery.
