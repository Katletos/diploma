import numpy as np

def read_pd_from_file_limited(filename: str) -> np.array:
    with open(filename, 'r') as file:
        lines = file.readlines()[:100]
        result = np.zeros(len(lines), dtype=np.float32)

        for i, l in enumerate(lines):
            result[i] = float(l)
        
        return result 



def read_pd_from_file(filename: str) -> np.array:
    with open(filename, 'r') as file:
        lines = file.readlines()
        result = np.zeros(len(lines), dtype=np.float32)

        for i, l in enumerate(lines):
            result[i] = float(l)
        
        return result 