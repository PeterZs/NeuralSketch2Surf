"""Utility functions shared by the standalone SwinUNETR implementation."""
import collections.abc
from itertools import repeat

def ensure_tuple_rep(t, repeat_nr):
    """Broadcast scalars to tuples while validating explicit sequences."""
    if isinstance(t, str):
        return (t,) * repeat_nr
    
    if isinstance(t, collections.abc.Sequence):
        if len(t) != repeat_nr:
            raise ValueError(f"Sequence length {len(t)} does not match repeat_nr {repeat_nr}.")
        return tuple(t)

    return tuple(repeat(t, repeat_nr))


def look_up_option(opt, supported, default="no_default", print_all_options=True):
    """Validate a string option against a small supported set."""

    if opt is None:
        if default != "no_default":
            return default
        return None
    
    if isinstance(opt, str):
        opt = opt.lower()
        
    if opt in supported:
        return opt
        
    raise ValueError(f"Option '{opt}' not in supported: {supported}")
