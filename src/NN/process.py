import numpy as np
from typing import Union, Sequence, Tuple, Type, Dict
#Num = Union[int, float]

def split_sequence(data: Sequence, n_snapshots: int):
	X, y = [], []
	
	for i in range(len(data)):
		# find the end of this pattern
		end_idx = i + n_snapshots
		# check if we are beyond the sequence
		if end_idx > len(data)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = data[i:end_idx], data[end_idx]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)
