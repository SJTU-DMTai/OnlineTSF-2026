"""Validate coarse-to-fine action lengths and per-round lookahead lengths."""

from .actions import positive_integer


def resolve_search_schedule(*, k: list[int] | tuple[int, ...] | None = None,
                            k_max: int | None = None, k_min: int | None = None,
                            h_tail: int | list[int] | tuple[int, ...] = 128) -> tuple[tuple[int, int], ...]:
    """Use explicit divisible steps, or halve k_max down to k_min (64..16 by default)."""
    if k is not None:
        if k_max is not None or k_min is not None:
            raise ValueError("specify search.k or search.k_max/k_min, not both")
        if not isinstance(k, (list, tuple)) or not k:
            raise ValueError("search.k must be a nonempty sequence of positive integers")
        steps = list(k)
    else:
        k_max = 64 if k_max is None else k_max
        k_min = 16 if k_min is None else k_min
        positive_integer(k_max, "search.k_max")
        positive_integer(k_min, "search.k_min")
        if k_max < k_min or k_max % k_min:
            raise ValueError("search.k_max must equal search.k_min * 2**n for an integer n >= 0")
        ratio = k_max // k_min
        if ratio & (ratio - 1):
            raise ValueError("search.k_max must equal search.k_min * 2**n for an integer n >= 0")
        steps = [k_max]
        while steps[-1] > k_min:
            steps.append(steps[-1] // 2)
    for step in steps:
        positive_integer(step, "search.k")
    for coarse, fine in zip(steps, steps[1:]):
        if coarse <= fine or coarse % fine:
            raise ValueError("search.k must strictly decrease, with each step dividing the preceding step")
    tails = list(h_tail) if isinstance(h_tail, (list, tuple)) else [h_tail] * len(steps)
    if len(tails) != len(steps):
        raise ValueError("search.h_tail must have one entry per search round")
    if any(isinstance(tail, bool) or not isinstance(tail, int) or tail < 0 for tail in tails):
        raise ValueError("search.h_tail must contain nonnegative integers")
    return tuple(zip(steps, tails))
