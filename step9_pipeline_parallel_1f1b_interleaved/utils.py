import torch
import random
import numpy as np
import builtins
import fcntl

def print(*args, is_print_rank=True, **kwargs):
    """ solves multi-process interleaved print problem """
    if not is_print_rank: return
    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            builtins.print(*args, **kwargs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)

def set_all_seed(seed):
    for module in [random, np.random]: module.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    
def to_readable_format(num, precision=3):
    num_str = str(num)
    length = len(num_str)
    
    def format_with_precision(main, decimal, suffix):
        if precision == 0:
            return f"{main}{suffix}"
        return f"{main}.{decimal[:precision]}{suffix}"
    
    if length > 12:  # Trillions
        return format_with_precision(num_str[:-12], num_str[-12:], 'T')
    elif length > 9:  # Billions
        return format_with_precision(num_str[:-9], num_str[-9:], 'B')
    elif length > 6:  # Millions
        return format_with_precision(num_str[:-6], num_str[-6:], 'M')
    elif length > 3:  # Thousands
        return format_with_precision(num_str[:-3], num_str[-3:], 'K')
    else:
        return num_str