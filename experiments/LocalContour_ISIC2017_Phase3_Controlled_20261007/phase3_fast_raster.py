"""JIT of the existing pixel-centre even/odd raster rule, with identical tolerances."""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _fill(polygon, h, w):
    result = np.zeros((h, w), dtype=np.bool_)
    n = len(polygon)
    if n < 3: return result
    xmin = max(0, int(np.ceil(np.min(polygon[:, 0]))))
    xmax = min(w-1, int(np.floor(np.max(polygon[:, 0]))))
    ymin = max(0, int(np.ceil(np.min(polygon[:, 1]))))
    ymax = min(h-1, int(np.floor(np.max(polygon[:, 1]))))
    for y in range(ymin, ymax+1):
        for x in range(xmin, xmax+1):
            inside = False
            on_edge = False
            for i in range(n):
                ax = polygon[i, 0]; ay = polygon[i, 1]
                j = 0 if i == n-1 else i+1
                bx = polygon[j, 0]; by = polygon[j, 1]
                dx = bx-ax; dy = by-ay
                rx = x-ax; ry = y-ay
                length2 = dx*dx+dy*dy
                if length2 > 1e-24:
                    projection = (rx*dx+ry*dy)/length2
                    cross = rx*dy-ry*dx
                    if abs(cross) <= 1e-8*max(1.0, np.sqrt(length2)) and projection >= -1e-10 and projection <= 1+1e-10:
                        on_edge = True
                crossing = (ay > y) != (by > y)
                if by != ay:
                    intercept = ax+(y-ay)*dx/dy
                    if crossing and x < intercept: inside = not inside
            result[y, x] = inside or on_edge
    return result


def rasterize_exact(points, shape):
    return _fill(np.asarray(points, dtype=np.float64), int(shape[0]), int(shape[1]))
