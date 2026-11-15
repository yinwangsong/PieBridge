"""Bounded, epoch-local frozen-feature planning; classifier order stays fixed."""


def feature_group(jobs, next_update, buffered, capacity, lookahead, group_sizes,
                  costs, budget, allow_reorder):
    """Return original batch IDs sharing a proxy, or [] if none fits.

    Costs are indexed by (proxy, number of logical batches). One buffer slot is
    reserved for the oldest missing batch so speculative work cannot deadlock it.
    No losses, future epochs, or unobserved execution durations are inspected.
    """
    available = capacity - len(buffered)
    if available <= 0:
        return []
    candidates = {}
    for index in range(next_update, min(len(jobs), next_update + lookahead)):
        if index not in buffered:
            candidates.setdefault(jobs[index]['proxy'], []).append(index)
    if not candidates:
        return []
    first = next(index for index in range(next_update, len(jobs)) if index not in buffered)
    head_proxy = jobs[first]['proxy']
    if not allow_reorder:
        # FIFO production can merge only consecutive same-proxy batches.
        consecutive = []
        for index in range(first, min(len(jobs), next_update + lookahead)):
            if index in buffered:
                continue
            if jobs[index]['proxy'] != head_proxy:
                break
            consecutive.append(index)
        candidates = {head_proxy: consecutive}
    order = [head_proxy] + ([p for p in candidates if p != head_proxy] if allow_reorder else [])
    choices = []
    for proxy in order:
        indices = candidates[proxy]
        includes_head = indices[0] == next_update
        free = available if includes_head or next_update in buffered else available - 1
        valid = [size for size in group_sizes
                 if size <= min(free, len(indices)) and costs[(proxy, size)] <= budget]
        if not valid:
            continue
        size = max(valid)
        if proxy == head_proxy:
            return indices[:size]
        choices.append((costs[(proxy, size)], -indices[0], indices[:size]))
    return max(choices)[2] if choices else []
