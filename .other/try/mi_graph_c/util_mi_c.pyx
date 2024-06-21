import numpy as np
cimport numpy as np
from cython.parallel import prange
from cpython cimport array
# from cython cimport view

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
DTYPE_int = np.int32


def bins_fd(np.ndarray[DTYPE_t, ndim=1] v not None):
    cdef:
        int v_len = len(v)
        double q_4[4]
        double iqr, bin_width, fd_float
        int num_bins
        double q_40, q_41, q_42, q_43

    np.percentile(v, [0, 25, 75, 100], overwrite_input=True, out=q_4)
    q_40, q_41, q_42, q_43 = q_4[0], q_4[1], q_4[2], q_4[3]
    iqr = q_42 - q_41
    bin_width = 2.0 * iqr * (v_len ** (-1.0 / 3.0))
    fd_float = (q_43 - q_40) / bin_width
    num_bins = <int>np.ceil(fd_float).item()
    if fd_float < 4:
        num_bins = 4
    return num_bins, q_40, q_43

def histogram2d_fast(
    np.ndarray[DTYPE_t, ndim=1] v1 not None,
    np.ndarray[DTYPE_t, ndim=1] v2 not None,
    int nbins_v1,
    int nbins_v2,
    DTYPE_t min_v1,
    DTYPE_t min_v2,
    DTYPE_t max_v1,
    DTYPE_t max_v2
):
    cdef:
        np.ndarray[DTYPE_t, ndim=2] hist2d = np.zeros((nbins_v1, nbins_v2), dtype=DTYPE)
        DTYPE_t bin_size_v1, bin_size_v2
        int i, v1_idx, v2_idx
        DTYPE_t v1i, v2i

    bin_size_v1 = (max_v1 - min_v1) / nbins_v1
    bin_size_v2 = (max_v2 - min_v2) / nbins_v2

    for i in range(len(v1)):
        v1i = v1[i]
        v2i = v2[i]
        v1_idx = <int>((v1i - min_v1) / bin_size_v1)
        v2_idx = <int>((v2i - min_v2) / bin_size_v2)

        if v1_idx >= nbins_v1:
            v1_idx = nbins_v1 - 1
        if v2_idx >= nbins_v2:
            v2_idx = nbins_v2 - 1

        hist2d[v1_idx, v2_idx] += 1

    hist2d /= len(v1)
    return hist2d

def mi_fd(np.ndarray[DTYPE_t, ndim=1] v1 not None, np.ndarray[DTYPE_t, ndim=1] v2 not None, bint normalized=True) -> float:
    cdef:
        int v1_n_bins, v2_n_bins
        double v1_min, v1_max, v2_min, v2_max
        np.ndarray[DTYPE_t, ndim=2] p_v1v2
        np.ndarray[DTYPE_t, ndim=1] p_v1, p_v2
        double mi_value, mi_best

    v1_n_bins, v1_min, v1_max = bins_fd(v1)
    v2_n_bins, v2_min, v2_max = bins_fd(v2)
    p_v1v2 = histogram2d_fast(v1, v2, v1_n_bins, v2_n_bins, v1_min, v2_min, v1_max, v2_max)
    p_v1 = np.sum(p_v1v2, axis=1)
    p_v2 = np.sum(p_v1v2, axis=0)
    mi_value = np.sum(p_v1v2 * np.log2(np.add(p_v1v2, 1e-31) / (p_v1[:, None] * p_v2[None, :])))
    mi_best = np.log2(np.sqrt(v1_n_bins * v2_n_bins))
    if normalized:
        mi_value = mi_value / mi_best
    return mi_value


def gen_windows_and_slides(
        int num_sample,
        slide_step,
        float width_range_low,
        float width_range_high,
        float width_step=0.1,
    ):
    """
    Generates sliding windows with changeable size.

    Args:
        num_sample: Total number of samples.
        slide_step: Step size for sliding windows (integer or fraction of total samples).
        width_range_low: Lower bound of window width range (fraction of total samples).
        width_range_high: Upper bound of window width range (fraction of total samples).
        width_step: Step size for window width (fraction of total samples).
    """
    cdef long len_step, width_min, width_max, width_stepL, num_width, i, n, start, end, sum_move
    cdef np.ndarray[long, ndim=1] widths, num_move
    cdef np.ndarray[long, ndim=1] pos_width, pos_move
    cdef np.ndarray[long, ndim=2] dfw

    # Calculate step length based on data type
    if isinstance(slide_step, int):
        len_step = slide_step
    else:
        len_step = int(np.ceil(num_sample * slide_step))

    # Convert width range and step to integers
    width_min = long(round(width_range_low * num_sample))
    width_max = long(round(width_range_high * num_sample))
    width_stepL = long(round(width_step * num_sample))
    if width_stepL < 1:
        width_stepL = 1

    # Calculate number of widths and corresponding values
    num_width = int((width_max - width_min) // width_stepL) + 1
    widths = np.array([width_min + i * width_stepL for i in range(num_width)])

    # Calculate number of moves for each width
    num_move = np.array([(num_sample - w) // len_step + 1 for w in widths])

    # Adjust last slide to cover remaining samples
    for i in range(len(num_move) - 1, -1, -1):
        if num_move[i] > 0 and (num_sample - widths[i]) % len_step > 0:
            num_move[i] += 1

    # Calculate total number of moves and prepare data structures
    sum_move = np.sum(num_move)
    pos_width = np.zeros(sum_move, dtype=int)
    pos_move = np.zeros(sum_move, dtype=int)

    # Fill in positions and moves for each window
    start = 0
    for i, n in enumerate(num_move):
        end = start + n
        pos_width[start:end] = widths[i]
        pos_move[start:end] = np.arange(n, dtype=int)
        start = end

    # Calculate start and end positions for each window
    pos_end = pos_width + pos_move * len_step
    pos_end = np.minimum(pos_end, num_sample)
    pos_sta = pos_end - widths + 1

    dfw = np.vstack((pos_sta, pos_end)).T
    return dfw


def std_each_window(np.ndarray[DTYPE_t, ndim=1] vec_in not None, np.ndarray[DTYPE_t, ndim=2] dfw not None) -> np.ndarray:
    cdef:
        int n_windows = dfw.shape[1]
        np.ndarray[DTYPE_t, ndim=1] vec_out = np.zeros(n_windows, dtype=DTYPE)
        int i, start, end

    for i in range(n_windows):
        start = <int>dfw[0, i]
        end = <int>dfw[1, i]
        vec_out[i] = np.std(vec_in[start:end])

    return vec_out

def slide_win_1d_sd(
    np.ndarray[DTYPE_t, ndim=1] vec_in not None,
    bint need_sort = True,
    slide_step = 0.1,
    float win_widths_range_low = 0.8,
    float win_widths_range_high = 0.98,
    float win_width_step = 0.1,
    float threshold = 0.05,
):
    cdef:
        np.ndarray[DTYPE_t, ndim=1] vec_in_s
        np.ndarray[DTYPE_t, ndim=2] dfw
        np.ndarray[DTYPE_t, ndim=1] vec_out
        bint is_save

    vec_in_s = vec_in
    if need_sort:
        vec_in_s = np.sort(vec_in_s)
    # Generate sliding windows
    dfw = gen_windows_and_slides(vec_in_s.shape[0], slide_step, win_widths_range_low, win_widths_range_high, win_width_step)

    vec_out = std_each_window(vec_in_s, dfw)
    is_save = vec_out.max() > threshold
    return is_save


def slide_win_2d_pcc(
    np.ndarray[DTYPE_t, ndim=1] v1 not None,
    np.ndarray[DTYPE_t, ndim=1] v2 not None,
    slide_step = 0.1,
    float win_widths_range_low = 0.8,
    float win_widths_range_high = 0.98,
    float win_width_step = 0.1,
) -> float:
    # Check input dimensions
    if v1.shape[0] != v2.shape[0]:
        raise ValueError('Input arrays must have the same number of samples.')
    
    # Generate sliding windows
    dfw = gen_windows_and_slides(v1.shape[0], slide_step, win_widths_range_low, win_widths_range_high, win_width_step)

    # Apply algorithm to each window
    cdef:
        int n_windows = dfw.shape[1]
        np.ndarray[DTYPE_t, ndim=1] vec_out = np.zeros(n_windows, dtype=DTYPE)
        int i, start, end

    for i in range(n_windows):
        start = <int>dfw[0, i]
        end = <int>dfw[1, i]
        vec_out[i] = np.corrcoef(v1[start:end], v2[start:end])[0,1]
    
    return vec_out.max()


def pcc_optimal(
    np.ndarray[DTYPE_t, ndim=1] v1 not None,
    np.ndarray[DTYPE_t, ndim=1] v2 not None,
    bint need_sort = True,
    slide_step = 0.1,
    float win_widths_range_low = 0.8,
    float win_widths_range_high = 0.98,
    float win_width_step = 0.05,
) -> float:
    if need_sort:
        sort_idx = np.argsort(v1)
        v1 = v1[sort_idx]
        v2 = v2[sort_idx]
    
    local_optimal = slide_win_2d_pcc(v1, v2, slide_step, win_widths_range_low, win_widths_range_high, win_width_step)
    global_value = np.corrcoef(v1, v2)[0,1]
    best_value = np.maximum(local_optimal, global_value)

    return best_value


def slide_win_2d_mutualinfo(
    np.ndarray[DTYPE_t, ndim=1] v1 not None,
    np.ndarray[DTYPE_t, ndim=1] v2 not None,
    slide_step = 0.1,
    float win_widths_range_low = 0.8,
    float win_widths_range_high = 0.98,
    float win_width_step = 0.1,
) -> float:
    # Check input dimensions
    if v1.shape[0] != v2.shape[0]:
        raise ValueError('Input arrays must have the same number of samples.')
    
    # Generate sliding windows
    dfw = gen_windows_and_slides(v1.shape[0], slide_step, win_widths_range_low, win_widths_range_high, win_width_step)

    # Apply algorithm to each window
    cdef:
        int n_windows = dfw.shape[1]
        np.ndarray[DTYPE_t, ndim=1] vec_out = np.zeros(n_windows, dtype=DTYPE)
        int i, start, end

    for i in range(n_windows):
        start = <int>dfw[0, i]
        end = <int>dfw[1, i]
        vec_out[i] = mi_fd(v1[start:end], v2[start:end])
    
    return vec_out.max()


def mi_optimal(
    np.ndarray[DTYPE_t, ndim=1] v1 not None,
    np.ndarray[DTYPE_t, ndim=1] v2 not None,
    bint need_sort = True,
    slide_step = 0.1,
    float win_widths_range_low = 0.8,
    float win_widths_range_high = 0.98,
    float win_width_step = 0.05,
) -> float:
    if need_sort:
        sort_idx = np.argsort(v1)
        v1 = v1[sort_idx]
        v2 = v2[sort_idx]
    
    local_optimal = slide_win_2d_mutualinfo(v1, v2, slide_step, win_widths_range_low, win_widths_range_high, win_width_step)
    global_value = mi_fd(v1, v2)
    mi_max = np.maximum(local_optimal, global_value)

    return mi_max


def iterate_feature_pairs_mt_mi(
        np.ndarray[double, ndim=2] matin not None,
        np.ndarray[long, ndim=2] feature_pairs not None
):
    cdef int n_pairs = feature_pairs.shape[0]
    cdef np.ndarray[double, ndim=1] vec_out = np.zeros(n_pairs, dtype=np.float64)
    cdef int i, n_features
    cdef double[:, :] matin_view = matin
    cdef long[:, :] feature_pairs_view = feature_pairs

    for i in range(n_pairs):
        vec_out[i] = mi_optimal(matin_view[:, feature_pairs_view[i, 0]], matin_view[:, feature_pairs_view[i, 1]])

    return vec_out, feature_pairs
