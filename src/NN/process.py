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

def get_snapshots_time(snapshots_dir, savepath: str = '', savename: str = '', save: bool = False) -> list:
    """
    Collect number of snapshots from snapshots_dir, return array containing it.
    """

    print("Getting snapshots....")

    snapshot_list = [float(i) for i in os.listdir(snapshots_dir)]
    # snapshot_list = [int(num) if float(num).is_integer() else float(num) for num in os.listdir(snapshots_dir)]
    snapshot_arr = np.sort(np.asarray(snapshot_list))

    snapshot_list = [int(x) if x == int(x) else x for x in snapshot_arr]

    # print(snapshot_arr)

    if save:
        os.makedirs(savepath, exist_ok=True)
        np.savetxt(savepath + savename + '.txt', snapshot_arr)

    return snapshot_list