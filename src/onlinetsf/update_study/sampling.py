"""Decision-origin sampling, independent of replay and action evaluation."""

import numpy as np


def sample_positions(start: int, stop: int, count: int, *,
                     segments: int | None = None, seed: int = 0) -> list[int]:
    """Sample distinct clocks in [start, stop), returned in time order.

    Split the interval into nearly equal consecutive segments and divide the
    sample budget evenly between them. Remainders go to the earlier segments.
    Each segment samples uniformly without replacement. By default there is one
    segment per sample. Segment boundaries constrain origins, not future labels.
    """
    if start < 0 or stop <= start:
        raise ValueError("sampling requires 0 <= start < stop")
    if not 0 < count <= stop - start:
        raise ValueError("positions must be positive and not exceed the available clocks")
    segments = count if segments is None else segments
    if not 0 < segments <= count:
        raise ValueError("sampling-segments must be between 1 and positions")
    if seed < 0:
        raise ValueError("sampling-seed must be nonnegative")

    width, extra_width = divmod(stop - start, segments)
    quota, extra_quota = divmod(count, segments)
    rng = np.random.default_rng(seed)
    clocks = []
    left = start
    for segment in range(segments):
        size = width + (segment < extra_width)
        samples = quota + (segment < extra_quota)
        clocks.extend((left + rng.choice(size, size=samples, replace=False)).tolist())
        left += size
    return sorted(clocks)
