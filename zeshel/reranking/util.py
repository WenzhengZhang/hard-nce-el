import math


def get_window(prefix, mention, suffix, max_size):
    if len(mention) >= max_size:
        window = mention[:max_size]
        return window, 0, len(window) - 1

    leftover = max_size - len(mention)
    leftover_half = int(math.ceil(leftover / 2))

    if len(prefix) >= leftover_half:
        prefix_len = leftover_half if len(suffix) >= leftover_half else \
                     leftover - len(suffix)
    else:
        prefix_len = len(prefix)

    prefix = prefix[-prefix_len:]  # Truncate head of prefix
    window = prefix + mention + suffix
    window = window[:max_size]  # Truncate tail of suffix

    mention_start_index = len(prefix)
    mention_end_index = len(prefix) + len(mention) - 1

    return window, mention_start_index, mention_end_index
