from matplotlib import ticker
from matplotlib.ticker import FuncFormatter, ScalarFormatter
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import seaborn as sns
from scipy.stats import norm

def calc_metrics(pds: list, number_of_responses, pd_deviation_sigma) -> list:
    unif = __analyze_unif(pds[0], number_of_responses, pd_deviation_sigma)
    stable = __analyze_stable(pds, number_of_responses, pd_deviation_sigma)
    rel = __analyze_reliability(pds, number_of_responses, pd_deviation_sigma)
    uniq = __analyze_uniqueness(pds, number_of_responses, pd_deviation_sigma)

    return [unif, stable, rel, uniq]

def format_func(value, tick_number):
    return f'{value:g}'  # Убирает научную нотацию

def plot_metrics(metrics: list):
    figure, axs = plt.subplots(2,2)
    figure.set_size_inches(20, 15)
    axs = axs.ravel()
    
    for i in range(len(metrics)):
        x = metrics[i]
        sns.histplot(x, bins=20, kde=False, stat='density', ax=axs[i], 
                    color="green", edgecolor='black')
        
        # Нормальное распределение
        mean = np.mean(x)
        std_dev = np.std(x)
        x_values = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, 100)
        p = norm.pdf(x_values, mean, std_dev)
        
        # Нанесение кривой нормального распределения
        axs[i].plot(x_values, p, color='red', linewidth=2, label='Normal distribution')
        
        # Настройки графика
        fontsize=26
        axs[i].set_title(["Uniformity", "Stable", "Reliability","Uniqueness"][i], fontsize=18)
        axs[i].set_xlabel('Values', fontsize=fontsize)
        axs[i].set_ylabel('Probability density', fontsize=fontsize)
        axs[i].tick_params(axis='both', labelsize=16)
        axs[i].yaxis.set_major_formatter(FuncFormatter(format_func))
        axs[i].xaxis.set_major_formatter(FuncFormatter(format_func))
        axs[i].grid()
        axs[i].legend(fontsize=fontsize)

    plt.tight_layout()
    plt.show()




def __analyze_unif(pd: np.array, number_of_responses: int, pd_deviation_sigma: float) -> np.array:
    responses = []
    for _ in range(number_of_responses):
        pairs = __select_pairwise(pd)
        bit_vector = __calc_bit_vector(pairs, pd_deviation_sigma)
        responses.append(bit_vector)

    total = np.zeros(len(responses), dtype=np.float32)
    for i in range(len(responses)):
        # total[i] = calc_uniformity(responses[i])
        total[i] = __calc_uniformity_normalized(responses[i])

    return total

def __analyze_stable(pds: list, number_of_responses: int, pd_deviation_sigma: float) -> np.array:
    res = np.zeros(len(pds), dtype=np.float32)

    for i, pd in enumerate(pds):
        responses = [ __calc_bit_vector(__select_pairwise(pd), pd_deviation_sigma) for _ in range(number_of_responses) ]
        res[i] = __calc_stable(responses)
        print(res[i])

    return res

def __analyze_reliability(pds: list, number_of_responses: int, pd_deviatio_sigma: float) -> np.array:
    total = np.zeros(len(pds), dtype=np.float32)

    for cur_iter, pd in enumerate(pds):
        pairs = __select_pairwise(pd)
        responses = [__calc_bit_vector(pairs, pd_deviatio_sigma) for _ in range(number_of_responses)]
        total[cur_iter] = __calc_reliability(responses)

    return total

def __analyze_uniqueness(pds: list, number_of_responses: int, pd_deviation_sigma: float) -> np.array:
    total = []
    for _ in range(number_of_responses):
        fpga_resps= [ __calc_bit_vector(__select_pairwise(pd), pd_deviation_sigma) for pd in pds ]
        result = __calc_uniquness(fpga_resps)
        total.append(result)

    return np.array(total)





def __select_pairwise(input_array: np.array) -> np.array:
    x = np.array(np.meshgrid(input_array, input_array)).T.reshape(-1, 2)
    y = np.delete(x, np.arange(0, len(input_array)**2, len(input_array)+1), axis=0)
    return y

def __calc_bit_vector(pairs: np.array, pd_deviation_sigma: int) -> np.array:
    kek = np.hsplit(pairs, 2)
    kek0 = kek[0].flatten() * __get_deviation(len(kek[0]), pd_deviation_sigma)
    kek1 = kek[1].flatten() * __get_deviation(len(kek[0]), pd_deviation_sigma)
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
    print(responses)
    s = np.add.reduce(responses)
    print(s)
    mask = (s != 0) & (s != len(responses))
    print(mask)
    # mask = s != len(responses)
    unique, counts = np.unique(mask, return_counts=True)
    print(unique)
    print(counts)
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