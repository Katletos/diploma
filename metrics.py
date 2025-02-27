import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from distfit import distfit

def calc_metrics(pds: list, number_of_responses) -> list:
    unif = __analyze_unif(pds[0], number_of_responses)
    # stable = __analyze_stable(pd, number_of_responses)
    stable = np.ones(25)
    rel = __analyze_reliability(pds, number_of_responses)
    uniq = __analyze_uniqueness(pds, number_of_responses)

    return [unif, stable, rel, uniq]

def plot_metrics(metrics: list):
    figure, axs = plt.subplots(2,2)
    figure.set_size_inches(20, 15)
    axs = axs.ravel()
    
    dfit = distfit(distr="norm")

    dfit.fit_transform(metrics[0])
    dfit.plot(ax=axs[0])
    axs[0].set_title("Uniformity")

    dfit.fit_transform(metrics[1])
    dfit.plot(ax=axs[1])
    axs[1].set_title("Stable")

    dfit.fit_transform(metrics[2])
    dfit.plot(ax=axs[2])
    axs[2].set_title("Reliability")

    dfit.fit_transform(metrics[3])
    dfit.plot(ax=axs[3])
    axs[3].set_title("Uniqueness")



def __analyze_unif(pd: np.array, number_of_responses: int) -> np.array:
    responses = []
    for _ in range(number_of_responses):
        pairs = __select_pairwise(pd)
        bit_vector = __calc_bit_vector(pairs)
        responses.append(bit_vector)

    total = np.zeros(len(responses), dtype=np.float32)
    for i in range(len(responses)):
        # total[i] = calc_uniformity(responses[i])
        total[i] = __calc_uniformity_normalized(responses[i])

    return total

def __analyze_stable(pds: np.array, number_of_responses: int) -> np.array:
    res = []
    for _ in range():

        responses = []
        for _ in range(number_of_responses):
            pairs = __select_pairwise(pd)
            bit_vector = __calc_bit_vector(pairs)
            responses.append(bit_vector)
        res.append(__calc_stable(responses))

def __analyze_reliability(pds: list, number_of_responses: int) -> np.array:
    total = np.zeros(len(pds), dtype=np.float32)

    for cur_iter, pd in enumerate(pds):
        responses = []
        pairs = __select_pairwise(pd)

        for i in range(number_of_responses):
            bit_vector = __calc_bit_vector(pairs)
            responses.append(bit_vector)

        total[cur_iter] = __calc_reliability(responses)

    return total

def __analyze_uniqueness(pds: list, number_of_responses: int) -> np.array:
    total = []
    fpga_num = len(pds)
    for _ in range(number_of_responses):
        fpga_resp = []
        for cur_iter, pd in enumerate(pds):
            pairs = __select_pairwise(pd)
            bit_vector = __calc_bit_vector(pairs)
            fpga_resp.append(bit_vector)

        s = 0
        for i in range(0, fpga_num - 1):
            for j in range(i + 1, fpga_num):
                s += distance.hamming(fpga_resp[i], fpga_resp[j])

        result = 2 / (fpga_num * (fpga_num - 1)) * s 
        total.append(result)

    return np.array(total)



def __select_pairwise(input_array: np.array) -> np.array:
    x = np.array(np.meshgrid(input_array, input_array)).T.reshape(-1, 2)
    y = np.delete(x, np.arange(0, len(input_array)**2, len(input_array)+1), axis=0)
    return y

def __calc_bit_vector(pairs: np.array) -> np.array:
    kek = np.hsplit(pairs, 2)
    kek0 = kek[0].flatten() * __get_deviation(len(kek[0]))
    kek1 = kek[1].flatten() * __get_deviation(len(kek[0]))
    result = (kek0 > kek1).astype(int)
    return result

def __get_deviation(size: int) -> np.array:
    PD_DEVIATION_SIGMA = 0.05
    return 1 + np.random.normal(0, PD_DEVIATION_SIGMA, size=size)




def __calc_uniformity_normalized(bit_vector: np.array) -> float:
    unique, counts = np.unique(bit_vector, return_counts=True)
    uniformity = 1 - 2 * abs(0.5 - counts[1] / bit_vector.size)
    return uniformity

def __calc_uniformity(bit_vector: np.array) -> float:
    unique, counts = np.unique(bit_vector, return_counts=True)
    uniformity = counts[1] / bit_vector.size
    return uniformity

def __calc_stable(responses: list) -> float:
    s = np.add.reduce(responses)
    mask = (s != 0) & (s != len(responses))
    unique, counts = np.unique(mask, return_counts=True)

    if len(counts) == 1:
        return 1
    else:
        return 1 - counts[1] / len(responses)

def __calc_reliability(responses: list) -> float:
    ref = responses[0]
    n = len(responses)
    s = 0

    for i in range(1, len(responses)):
        s += distance.hamming(responses[i], ref)

    return 1 - 1 / n * s

