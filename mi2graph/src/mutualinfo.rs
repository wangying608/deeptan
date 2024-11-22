use crate::slidingwindow::init_windows_from_ratio;
use crate::sortf64::{get_sort_indices_vecf64, sort_vec_f64, sort_vecs_by_first};
use ndarray::prelude::*;
use rayon::prelude::*;
use std::error::Error;
// use std::io::ErrorKind;

/// Iterates over all feature pairs and applies the mutual information algorithm.
/// + Accepts a 2D array of f64 values.
/// + Returns a vector of mutual information values and a 2D array of feature pairs which is composed by features' indices.
///
/// Note:
/// + Each row in input `data` is a feature vector.
/// + Each row in input `feat_pairs` is a feature pair.
pub fn iter_feat_pairs_mi(
    data: &Array2<f64>,
    ratio_max_window: f64,
    ratio_min_window: f64,
    ratio_step_window: f64,
    ratio_step_sliding: f64,
    sort_results: bool,
) -> (Array1<f64>, Array2<i64>) {
    let n_feat = data.nrows();
    let n_samp = data.ncols();

    // Init sliding windows.
    let sld_windows = init_windows_from_ratio(
        n_samp,
        ratio_min_window,
        ratio_max_window,
        ratio_step_window,
        ratio_step_sliding,
    );

    //
    let n_pairs = n_feat * (n_feat - 1) / 2;
    // Initialize a vector to store mutual information values. (size = n_pairs)
    let mut mi_vec: Vec<f64> = vec![0.0; n_pairs];

    // Initialize a 2D array to store feature pairs.
    let mut feat_pairs = Array2::zeros((n_pairs, 2));
    let mut pos_v: usize = 0;
    for i in 0..n_feat {
        for j in i + 1..n_feat {
            feat_pairs[[pos_v, 0]] = i as i64;
            feat_pairs[[pos_v, 1]] = j as i64;
            pos_v += 1;
        }
    }
    assert_eq!(pos_v, n_pairs);
    // Iterate over all feature pairs.
    mi_vec.par_iter_mut().enumerate().for_each(|(i, mi)| {
        // Extract feature values.
        let feat_1 = data.select(Axis(0), &[feat_pairs[[i, 0]] as usize]);
        let feat_2 = data.select(Axis(0), &[feat_pairs[[i, 1]] as usize]);

        // Calculate mutual information.
        *mi = mi_optimal(&feat_1.into_raw_vec(), &feat_2.into_raw_vec(), &sld_windows);
    });

    // Sort the vector of mutual information values in descending order. feat_pairs are also sorted.
    if sort_results {
        let (mi_vec_sorted, feat_pairs_sorted) = sort_mi_results(&mi_vec, &feat_pairs);
        mi_vec = mi_vec_sorted;
        feat_pairs = feat_pairs_sorted;
    }

    // Return the vector of mutual information values.
    let nda_mi = Array1::from(mi_vec);
    (nda_mi, feat_pairs)
}

/// Sort mutual information values in descending order.
/// `feat_pairs` are also sorted.
fn sort_mi_results(mi_vec: &Vec<f64>, feat_pairs: &Array2<i64>) -> (Vec<f64>, Array2<i64>) {
    // Get descending indices.
    let mut sort_idx = get_sort_indices_vecf64(mi_vec);
    sort_idx.reverse();

    let mut feat_pairs_sorted = feat_pairs.clone();
    let mut mi_vec_sorted = mi_vec.clone();
    // Sort mutual information values and feature pairs.
    for (i, idx) in sort_idx.iter().enumerate() {
        mi_vec_sorted[i] = mi_vec[*idx];
        feat_pairs_sorted.row_mut(i).assign(&feat_pairs.row(*idx));
    }

    (mi_vec_sorted, feat_pairs_sorted)
}

/// Calculates the optimal mutual information between two features, using _DYNAMIC_ sliding window method.
///
/// + What's "_DYNAMIC_":
///     1. min window size
///     2. max window size
///     3. step width of window size
///     4. step width of slide
///
/// + Accepts two 1D arrays of floating-point values and parameters of sliding window.
/// + Returns the optimal mutual information value.
pub fn mi_optimal(feat_1: &Vec<f64>, feat_2: &Vec<f64>, sliding_windows: &Vec<Vec<usize>>) -> f64 {
    // Sort feat_1 and feat_2 in ascending order based on the sortperm of feat_1.
    let (sorted_feat_1, sorted_feat_2) = sort_vecs_by_first(feat_1, feat_2);

    // Initialize a vector to store mutual information values.
    let n_windows = sliding_windows.len();
    let mut mi_vec: Vec<f64> = vec![0.0; n_windows];

    // Iterate over all sliding windows. PARALLELIZED.
    mi_vec.par_iter_mut().enumerate().for_each(|(i, mi)| {
        // Extract the sliding window.
        let window = &sliding_windows[i];
        let tmp_sta = window[0];
        let tmp_end = window[1];

        // Calculate the mutual information.
        *mi = mi_fd(
            &sorted_feat_1[tmp_sta..tmp_end].to_vec(),
            &sorted_feat_2[tmp_sta..tmp_end].to_vec(),
            true,
        );
    });

    // Find the maximum mutual information value.
    // let sort_indices = get_sort_indices_vecf64(&mi_vec);
    // let mut mi_opt = mi_vec[sort_indices[mi_vec.len() - 1]];
    let sorted_mi_vec = sort_vec_f64(&mi_vec);
    let mut mi_opt = sorted_mi_vec[sorted_mi_vec.len() - 1];

    // Calculate the MI for the complete data.
    let mi_complete = mi_fd(feat_1, feat_2, true);

    if mi_complete > mi_opt {
        mi_opt = mi_complete;
    }

    // Return the optimal mutual information value.
    mi_opt
}

/// Mutual information algorithm.
/// + Accepts two vectors of f64 values and a bool value for normalization.
/// + Returns the mutual information value between the two vectors. The default value of normalization is true.
///
/// How to determine the bins?
///
/// RectangularBinning (the adaptive partitioning approach): Freedman-Diaconis' rule (no assumption on the distribution).
fn mi_fd(feat_1: &Vec<f64>, feat_2: &Vec<f64>, normalized: bool) -> f64 {
    // Generate the bins.
    let Ok((n_bins_f1, bin_width_f1, quantiles_f1)) = bins_fd(feat_1) else {
        // return Err(Box::new(std::io::Error::new(
        //     std::io::ErrorKind::Other,
        //     "Failed to generate bins.",
        // )));
        return 0.0;
    };
    let Ok((n_bins_f2, bin_width_f2, quantiles_f2)) = bins_fd(feat_2) else {
        // return Err(Box::new(std::io::Error::new(
        //     std::io::ErrorKind::Other,
        //     "Failed to generate bins.",
        // )));
        return 0.0;
    };

    // Calculate the mutual information.
    let mi = hist2mi(
        feat_1,
        feat_2,
        n_bins_f1,
        n_bins_f2,
        bin_width_f1,
        bin_width_f2,
        &quantiles_f1,
        &quantiles_f2,
        normalized,
    );

    mi
}

/// RectangularBinning (the adaptive partitioning approach): Freedman-Diaconis' rule (no assumption on the distribution)
/// + Returns the number of bins, the bin width, and the quantiles.
fn bins_fd(vec_x: &Vec<f64>) -> Result<(usize, f64, Vec<f64>), Box<dyn Error>> {
    let sort_indices = get_sort_indices_vecf64(vec_x);
    let len_vec = vec_x.len();
    let len_vec_f64 = len_vec as f64;
    let qs = &[0.0, 0.25, 0.75, 1.0];
    let mut quantiles: Vec<f64> = vec![0.0; qs.len()];
    // Calculate the quantiles
    // for (i, q) in qs.iter().enumerate() {
    //     quantiles[i] = vec_x[sort_indices[((len_vec as f64 * q).round() as usize) - 1]];
    // }
    quantiles[1] = vec_x[sort_indices[((len_vec_f64 * 0.5).round() as usize) - 1]];
    quantiles[2] = vec_x[sort_indices[((len_vec_f64 * 0.75).round() as usize) - 1]];
    quantiles[0] = vec_x[sort_indices[0]];
    quantiles[3] = vec_x[sort_indices[len_vec - 1]];

    let iqr = quantiles[2] - quantiles[1];
    let bin_width = 2.0 * iqr / (len_vec_f64).cbrt();

    let n_bins = ((quantiles[3] - quantiles[0]) / bin_width).ceil() as usize;

    if n_bins < 3 {
        // panic!("The number of bins is less than 3.");
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "The number of bins is less than 3.",
        )));
    }

    if n_bins > len_vec {
        // panic!("The number of bins is greater than the length of the vector.");
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "The number of bins is greater than the length of the vector.",
        )));
    }

    Ok((n_bins, bin_width, quantiles))
}

/// Calculate the mutual information value between two vectors using the histogram method.
fn hist2mi(
    vec_1: &Vec<f64>,
    vec_2: &Vec<f64>,
    n_bins_v1: usize,
    n_bins_v2: usize,
    bin_width_v1: f64,
    bin_width_v2: f64,
    quantiles_v1: &Vec<f64>,
    quantiles_v2: &Vec<f64>,
    normalized: bool,
) -> f64 {
    let len_vec = vec_1.len();
    let mut hist2d: Array2<f64> = Array2::zeros((n_bins_v1, n_bins_v2));
    let mut p_v1: Array1<f64> = Array1::zeros(n_bins_v1);
    let mut p_v2: Array1<f64> = Array1::zeros(n_bins_v2);

    // Loop through all data points
    for i in 0..len_vec {
        let bin_v1 = (vec_1[i] - quantiles_v1[0]) / bin_width_v1;
        let bin_v2 = (vec_2[i] - quantiles_v2[0]) / bin_width_v2;
        let mut tmp_ind_bin_v1 = (bin_v1.round() as isize) - 1;
        let mut tmp_ind_bin_v2 = (bin_v2.round() as isize) - 1;
        if tmp_ind_bin_v1 < 0 {
            tmp_ind_bin_v1 = 0;
        }
        if tmp_ind_bin_v2 < 0 {
            tmp_ind_bin_v2 = 0;
        }
        let ind_bin_v1 = tmp_ind_bin_v1 as usize;
        let ind_bin_v2 = tmp_ind_bin_v2 as usize;

        hist2d[[ind_bin_v1, ind_bin_v2]] += 1.0;
        p_v1[ind_bin_v1] += 1.0;
        p_v2[ind_bin_v2] += 1.0;
    }

    // Apply the normalization to the histogram for each element of hist2d.
    let norm_hist2d = hist2d.mapv(|x| x / (len_vec as f64));
    let norm_p_v1 = p_v1.mapv(|x| x / (len_vec as f64));
    let norm_p_v2 = p_v2.mapv(|x| x / (len_vec as f64));

    // Calculate the mutual information.
    let mut mi = 0.0;

    for i in 0..n_bins_v1 {
        for j in 0..n_bins_v2 {
            let p_ij = norm_hist2d[[i, j]];
            if p_ij > 1e-7 {
                let p_v1_i = norm_p_v1[i];
                let p_v2_j = norm_p_v2[j];
                mi += p_ij * (p_ij / (p_v1_i * p_v2_j)).log2();
            }
        }
    }

    // Apply the normalization if needed.
    if normalized {
        let mi_max = (n_bins_v1 as f64 * n_bins_v2 as f64).sqrt().log2();
        mi /= mi_max;
    }

    mi
}
