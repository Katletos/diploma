import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from distfit import distfit

def calc_metrics(pds: list, interconnect_pd: list, number_of_responses, pd_deviation_sigma) -> list:
    unif = __analyze_unif(pds[0], interconnect_pd[0], number_of_responses, pd_deviation_sigma)
    stable = __analyze_stable(pds, interconnect_pd, number_of_responses, pd_deviation_sigma)
    rel = __analyze_reliability(pds, interconnect_pd, number_of_responses, pd_deviation_sigma)
    uniq = __analyze_uniqueness(pds, interconnect_pd, number_of_responses, pd_deviation_sigma)

    return [unif, stable, rel, uniq]

def plot_metrics(metrics: list):
    figure, axs = plt.subplots(1,3)
    figure.set_size_inches(15, 5)
    axs = axs.ravel()
    
    dfit = distfit(distr="norm")

    dfit.fit_transform(metrics[0])
    dfit.plot(ax=axs[0])
    axs[0].set_title("Uniformity")

    # dfit.fit_transform(metrics[1])
    # dfit.plot(ax=axs[1])
    # axs[1].set_title("Stability")

    dfit.fit_transform(metrics[1])
    dfit.plot(ax=axs[1])
    axs[1].set_title("Reliability")

    dfit.fit_transform(metrics[2])
    dfit.plot(ax=axs[2])
    axs[2].set_title("Uniqueness")




def __analyze_unif(pd: np.array, interconnect_pd: np.array, number_of_responses: int, pd_deviation_sigma: float) -> np.array:
    responses = []
    for _ in range(number_of_responses):
        pairs = __select_pairwise(pd)
        interconnect = __select_pairwise(interconnect_pd)
        bit_vector = __calc_bit_vector(pairs, interconnect, pd_deviation_sigma)
        responses.append(bit_vector)

    total = np.zeros(len(responses), dtype=np.float32)
    for i in range(len(responses)):
        # total[i] = calc_uniformity(responses[i])
        total[i] = __calc_uniformity_normalized(responses[i])

    return total

def __analyze_stable(pds: list, interconnect_pd: list, number_of_responses: int, pd_deviation_sigma: float) -> np.array:
    res = np.zeros(len(pds), dtype=np.float32)

    for i in range(len(pds)):
        responses = []
        for _ in range(number_of_responses):
            pairs = __select_pairwise(pds[i])
            inter = __select_pairwise(interconnect_pd[i])
            bit_vector = __calc_bit_vector(pairs, inter, pd_deviation_sigma)
            responses.append(bit_vector)

        res[i] = __calc_stable(responses)

    return res

def __analyze_reliability(pds: list, interconnect_pd: list, number_of_responses: int, pd_deviatio_sigma: float) -> np.array:
    total = np.zeros(len(pds), dtype=np.float32)

    for i in range(len(pds)):
        pairs = __select_pairwise(pds[i])
        inter = __select_pairwise(interconnect_pd[i])
        responses = [__calc_bit_vector(pairs, inter, pd_deviatio_sigma) for _ in range(number_of_responses)]
        total[i] = __calc_reliability(responses)

    return total

def __analyze_uniqueness(pds: list, interconnect_pd: list, number_of_responses: int, pd_deviation_sigma: float) -> np.array:
    total = np.zeros(number_of_responses, dtype=np.float32)

    for i in range(number_of_responses):
        fpga_resps = []
        for j in range(len(pds)):
            pairs = __select_pairwise(pds[j])
            inter = __select_pairwise(interconnect_pd[j])
            fpga_resps.append(__calc_bit_vector(pairs, inter, pd_deviation_sigma))

        total[i] = __calc_uniquness(fpga_resps)

    return np.array(total)





def __select_pairwise(input_array: np.array, ) -> np.array:
    x = np.array(np.meshgrid(input_array, input_array)).T.reshape(-1, 2)
    y = np.delete(x, np.arange(0, len(input_array)**2, len(input_array)+1), axis=0)
    return y

def __calc_bit_vector(pairs: np.array, interconnect: np.array, pd_deviation_sigma: int) -> np.array:
    kek = np.hsplit(pairs, 2)
    inter = np.hsplit(interconnect, 2)

    kek0 = (kek[0].flatten() + inter[0].flatten()) * __get_deviation(len(kek[0]), pd_deviation_sigma)
    kek1 = (kek[1].flatten() + inter[1].flatten()) * __get_deviation(len(kek[0]), pd_deviation_sigma)
    result = (kek0 > kek1).astype(int)
    return result

def __get_deviation(size: int, pd_deviation_sigma: float) -> np.array:
    return 1 + np.random.normal(0, pd_deviation_sigma, size=size)




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
    # mask = s != len(responses)
    unique, counts = np.unique(mask, return_counts=True)
    return 0 if len(counts) == 1 else 1 - counts[0] / len(s)

def __calc_reliability(responses: list) -> float:
    ref = responses[0]
    s = 0
    m = len(responses)

    for i in range(1, m):
        s += distance.hamming(responses[i], ref)

    return 1 - 1 / m * s

def __calc_uniquness(fpga_resps: list) -> float:
    n = len(fpga_resps)
    s = 0
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            s += distance.hamming(fpga_resps[i], fpga_resps[j])

    return 2 / (n * (n - 1)) * s 