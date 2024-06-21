import numpy as np
from numba import njit


@njit(cache=True)
def bins_fd(v: np.ndarray):
    '''
    RectangularBinning (the adaptive partitioning approach)
    Freedman-Diaconis' rule (no assumption on the distribution)
    '''
    v_len = len(v)
    # v_min, q1, q3, v_max
    q_4 = np.percentile(v, [0, 25, 75, 100])
    iqr = q_4[2] - q_4[1]
    bin_width = 2.0 * iqr * (v_len ** (-1.0 / 3.0))
    fd_float = (q_4[3] - q_4[0]) / bin_width
    num_bins = int(np.ceil(fd_float))
    if fd_float < 4:
        num_bins = 4
    return num_bins, q_4[0], q_4[3]


@njit(cache=True)
def njit_histogram2d(
        v1: np.ndarray,
        v2: np.ndarray,
        nbins_v1: int,
        nbins_v2: int,
        min_v1: float,
        min_v2: float,
        max_v1: float,
        max_v2: float,
    ):
    hist2d = np.zeros((nbins_v1, nbins_v2), dtype=np.float32)
    bin_size_v1 = (max_v1 - min_v1) / nbins_v1
    bin_size_v2 = (max_v2 - min_v2) / nbins_v2
    for i in range(len(v1)):
        v1_idx = int((v1[i] - min_v1) / bin_size_v1)
        v2_idx = int((v2[i] - min_v2) / bin_size_v2)
        if v1_idx >= nbins_v1:
            v1_idx = nbins_v1 - 1
        if v2_idx >= nbins_v2:
            v2_idx = nbins_v2 - 1
        hist2d[v1_idx, v2_idx] += 1
    hist2d /= len(v1)
    return hist2d


@njit(cache=True, fastmath=True)
def mi_fd(v1: np.ndarray, v2: np.ndarray, normalized: bool = True) -> float:
    v1_n_bins, v1_min, v1_max = bins_fd(v1)
    v2_n_bins, v2_min, v2_max = bins_fd(v2)

    # NumpPy implementation:
    # hist2d = np.histogram2d(v1, v2, bins=[v1_n_bins, v2_n_bins], range=[[v1_min, v1_max], [v2_min, v2_max]])
    # p_v1v2 = hist2d[0] / len(v1)

    # Numba implementation:
    p_v1v2 = njit_histogram2d(v1, v2, v1_n_bins, v2_n_bins, v1_min, v2_min, v1_max, v2_max)

    p_v1 = np.sum(p_v1v2, axis=1)
    p_v2 = np.sum(p_v1v2, axis=0)

    v1_m_v2t = p_v1.reshape(-1, 1) * p_v2.reshape(1, -1)
    mi_tmp = p_v1v2 * np.log2(np.add(p_v1v2, 1e-31) / v1_m_v2t)
    mi_value = np.sum(mi_tmp)

    mi_best = np.log2(np.sqrt(v1_n_bins * v2_n_bins))
    if normalized:
        mi_value = mi_value / mi_best
    return mi_value


@njit(cache=True)
def gen_windows_and_slides(
        num_sample: int,
        slide_step: int | float,
        width_range_low: float,
        width_range_high: float,
        width_step: float = 0.1,
    ):
    """
    Generates sliding windows with changeable size.

    Args:
        num_sample: Total number of samples.
        slide_step: Step size for sliding windows (integer or fraction of total samples).
        width_range: Range of window widths (must be within (0, 1)).
        width_step: Step size for window width (fraction of total samples).
    """

    # if width_range_high >= 1:
    #     raise ValueError("Range of window width must be within (0, 1)")
    # if width_range_low < 0:
    #     raise ValueError("Range of window width must be within (0, 1)")
    # if width_step <= 0 or width_step >= 1:
    #     raise ValueError("Step size for window width must be within (0, 1)")

    # Calculate step length based on data type
    if isinstance(slide_step, int):
        len_step = slide_step
    else:
        len_step = int(np.ceil(num_sample * slide_step))

    # Convert width range and step to integers
    # width_min, width_max = map(int, map(lambda x: x * num_sample, width_range))
    width_min = int(round(width_range_low * num_sample))
    width_max = int(round(width_range_high * num_sample))
    width_stepL = int(round(width_step * num_sample))
    if width_stepL < 1:
        width_stepL = 1

    # Calculate number of widths and corresponding values
    num_width = int((width_max - width_min) // width_stepL) + 1
    tmp_w_seq = range(num_width)
    widths = [width_min + i * width_stepL for i in tmp_w_seq]

    # Calculate number of moves for each width
    num_move = [int((num_sample - w) // len_step) + 1 for w in widths]

    # Adjust last slide to cover remaining samples
    for i in range(len(num_move) - 1, -1, -1):
        if num_move[i] > 0 and (num_sample - widths[i]) % len_step > 0:
            num_move[i] += 1

    # Calculate total number of moves and prepare data structures
    sum_move = sum(num_move)
    pos_width = [0] * sum_move
    pos_move = [0] * sum_move

    # Fill in positions and moves for each window
    start = 0
    for i, n in enumerate(num_move):
        end = start + n
        pos_width[start:end] = [widths[i]] * n
        pos_move[start:end] = [j for j in range(n)]
        start = end

    # Calculate start and end positions for each window
    pos_end = [w + m * len_step for w, m in zip(pos_width, pos_move)]
    pos_end = [min(num_sample, p) for p in pos_end]
    pos_sta = [p - w + 1 for p, w in zip(pos_end, pos_width)]

    # Create and return DataFrame
    # dfw = pd.DataFrame(dict(window_sta=pos_sta, window_end=pos_end, widths=pos_width, slides=pos_move))
    # generate dfw as numpy instead of dataframe
    # dfw = np.array([pos_sta, pos_end, pos_width, pos_move])
    dfw = np.array([pos_sta, pos_end])
    return dfw


@njit(cache=True)
def std_each_window(vec_in: np.ndarray, dfw: np.ndarray) -> np.ndarray:
    '''
    Apply algorithm to each window
    '''
    n_windows = dfw.shape[1]
    vec_out = np.zeros(n_windows)
    for i in range(n_windows):
        vec_out[i] = np.std(vec_in[dfw[0,i]:dfw[1,i]])
    return vec_out


@njit(cache=True)
def slide_win_1d_sd(
        vec_in: np.ndarray,
        need_sort: bool = True,
        slide_step: int | float = 0.1,
        win_widths_range_low: float = 0.8,
        win_widths_range_high: float = 0.98,
        win_width_step: float = 0.1,
        threshold: float = 0.05,
    ):
    '''
    Plead attention:
        vec_in must be sorted along the sortperm of vec_in.
    
    Applies a sliding window algorithm to a 1D numpy array,
    using a specified algorithm:
        [ standard deviation ].
    '''
    vec_in_s = vec_in
    if need_sort:
        vec_in_s = np.sort(vec_in_s)
    # Generate sliding windows
    dfw = gen_windows_and_slides(vec_in_s.shape[0], slide_step, win_widths_range_low, win_widths_range_high, win_width_step)

    vec_out = std_each_window(vec_in_s, dfw)

    # if get_optimal:
    #     max_std = vec_out.max()
    #     if get_bool:
    #         return max_std > threshold
    #     else:
    #         return max_std
    # else:
    #     return vec_out, dfw
    is_save: bool = vec_out.max() > threshold
    return is_save


@njit(cache=True)
def slide_win_2d_pcc(
        v1: np.ndarray,
        v2: np.ndarray,
        slide_step: int | float = 0.1,
        win_widths_range_low: float = 0.8,
        win_widths_range_high: float = 0.98,
        win_width_step: float = 0.1,
    ) -> float:
    '''
    Please attention:
        v1 and v2 must be sorted along the sortperm of v1 or v2.
    
    Applies a sliding window algorithm to a 2D numpy array, whose axes are vec_1 and vec_2,
    using a specified algorithm:
        [ Pearson correlation coefficient ].
    '''
    # Check input dimensions
    if v1.shape[0] != v2.shape[0]:
        raise ValueError('Input arrays must have the same number of samples.')
    
    # Generate sliding windows
    dfw = gen_windows_and_slides(v1.shape[0], slide_step, win_widths_range_low, win_widths_range_high, win_width_step)

    # Apply algorithm to each window
    n_windows = dfw.shape[1]
    vec_out = np.zeros(n_windows)

    for i in range(n_windows):
        vec_out[i] = np.corrcoef(v1[dfw[0,i]:dfw[1,i]], v2[dfw[0,i]:dfw[1,i]])[0,1]
    
    return vec_out.max()


@njit(cache=True)
def pcc_optimal(
        v1: np.ndarray,
        v2: np.ndarray,
        need_sort: bool = True,
        slide_step: int | float = 0.1,
        win_widths_range_low: float = 0.8,
        win_widths_range_high: float = 0.98,
        win_width_step: float = 0.05,
    ) -> float:
    '''
    Returns the optimal (global or local) value of PCC between two 1D numpy arrays.

    This function is an atom function for the parallel processing for all feature pairs.
    '''

    if need_sort:
        sort_idx = np.argsort(v1)
        v1 = v1[sort_idx]
        v2 = v2[sort_idx]
    
    local_optimal = slide_win_2d_pcc(v1, v2, slide_step, win_widths_range_low, win_widths_range_high, win_width_step)
    global_value = np.corrcoef(v1, v2)[0,1]
    best_value: float = np.maximum(local_optimal, global_value)

    return best_value


# def generate_feat_combinations(n: int):
#     '''
#     Generates all possible combinations of 2 features from n features.
#     '''
#     n_combinations = n * (n - 1) // 2
#     features_combinations = np.zeros((n_combinations, 2), dtype=int)
#     features = np.arange(n)
#     feat_combinations = combinations(features, 2)
#     i = 0
#     for ft1, ft2 in feat_combinations:
#         features_combinations[i, 0] = ft1
#         features_combinations[i, 1] = ft2
#         i += 1
#     return features_combinations


@njit(cache=True)
def slide_win_2d_mutualinfo(
        v1: np.ndarray,
        v2: np.ndarray,
        slide_step: int | float = 0.1,
        win_widths_range_low: float = 0.8,
        win_widths_range_high: float = 0.98,
        win_width_step: float = 0.1,
        # get_optimal: bool = True,
    ) -> float:
    '''
    Please attention:
        v1 and v2 must be sorted along the sortperm of v1 or v2.
    
    Applies a sliding window algorithm to a 2D numpy array, whose axes are vec_1 and vec_2,
    using a specified algorithm:
        [ mutual information ].
    '''
    # Check input dimensions
    if v1.shape[0] != v2.shape[0]:
        raise ValueError('Input arrays must have the same number of samples.')
    
    # Generate sliding windows
    dfw = gen_windows_and_slides(v1.shape[0], slide_step, win_widths_range_low, win_widths_range_high, win_width_step)

    # Apply algorithm to each window
    n_windows = dfw.shape[1]
    vec_out = np.zeros(n_windows)

    # if n_thread == 1:
    for i in range(n_windows):
        vec_out[i] = mi_fd(v1[dfw[0,i]:dfw[1,i]], v2[dfw[0,i]:dfw[1,i]])
    # if n_thread == "auto":
    #     n_thread = round(multiprocessing.cpu_count() * 0.9)
    #     vec_out = Parallel(n_jobs=n_thread)(delayed(mi_fd)(vec_1[row['window_sta']:row['window_end']], vec_2[row['window_sta']:row['window_end']]) for i, row in dfw.iterrows())
    #     vec_out = np.array(vec_out)
    
    # if get_optimal:
    #     return vec_out.max()
    # else:
    #     return vec_out, dfw
    return vec_out.max()


@njit(cache=True)
def mi_optimal(
        v1: np.ndarray,
        v2: np.ndarray,
        need_sort: bool = True,
        slide_step: int | float = 0.1,
        win_widths_range_low: float = 0.8,
        win_widths_range_high: float = 0.98,
        win_width_step: float = 0.05,
    ) -> float:
    '''
    Returns the optimal (global or local) value
    of mutual information between two 1D numpy arrays.

    This function is an atom function for the parallel processing for all feature pairs.
    '''

    if need_sort:
        sort_idx = np.argsort(v1)
        v1 = v1[sort_idx]
        v2 = v2[sort_idx]
    
    local_optimal = slide_win_2d_mutualinfo(v1, v2, slide_step, win_widths_range_low, win_widths_range_high, win_width_step)
    global_value = mi_fd(v1, v2)
    mi_max: float = np.maximum(local_optimal, global_value)

    return mi_max


@njit(cache=True)
def iterate_feature_pairs_mt_mi(
        matin: np.ndarray,
        feature_pairs: np.ndarray,
    ):
    '''
    Iterates over all feature pairs and applies the mutual information algorithm.
    '''
    # Parallel method 1
    # vec_out = Parallel(n_jobs=n_processes)(delayed(mi_optimal)(matin.copy()[:,feat1], matin.copy()[:,feat2]) for feat1, feat2 in feat_combinations)
    # parallel method doesn't works ?

    # Parallel method 2 ?
    
    n_pairs = feature_pairs.shape[0]
    vec_out = np.zeros(n_pairs)

    # for feat1, feat2 in feat_combinations:
    #     vec_out.append(mi_optimal(matin.copy()[:,feat1], matin.copy()[:,feat2]))
    
    for i in range(n_pairs):
        vec_out[i] = mi_optimal(matin.copy()[:,feature_pairs[i,0]], matin.copy()[:,feature_pairs[i,1]])
    # vec_out = Parallel(n_jobs=n_processes)(delayed(mi_optimal)(matin.copy()[:,feature_pairs[i,0]], matin.copy()[:,feature_pairs[i,1]]) for i in range(n_pairs))

    # Return MI values and feature pairs
    return vec_out, feature_pairs
    # return np.array(vec_out)
