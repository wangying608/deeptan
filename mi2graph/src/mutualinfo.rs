use crate::sortf64::{get_sort_indices_vecf64, get_sort_indices_vecf64_slice, sort_vec_f64};
use ndarray::prelude::*;
use rayon::prelude::*;
use std::error::Error;

/// Iterates over all feature pairs and applies the mutual information algorithm.
/// + Accepts a 2D array of f64 values.
/// + Returns a vector of mutual information values and a 2D array of feature pairs which is composed by features' indices.
///
/// Note:
/// + Each row in input `data` is a feature vector.
/// + Each row in input `feat_pairs` is a feature pair.
pub fn iter_feat_pairs_mi(
    data: &Array2<f64>,
    sliding_windows: &Vec<Vec<usize>>,
    features_sort_indices: &Vec<Vec<usize>>,
    sort_results: bool,
) -> (Array1<f64>, Array2<i64>) {
    let n_feat = data.nrows();

    // Calculate the number of feature pairs.
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
    // mi_vec.par_iter_mut().enumerate().for_each(|(i, mi)| {
    //     // Extract feature values.
    //     let feat_1 = &features[feat_pairs[[i, 0]] as usize];
    //     let feat_2 = &features[feat_pairs[[i, 1]] as usize];
    //     // Calculate mutual information.
    //     *mi = mi_optimal(feat_1, feat_2, &sld_windows);
    // });

    // Split the mi_vec into chunks for parallel processing.
    let chunk_size = (n_pairs + rayon::current_num_threads() - 1) / rayon::current_num_threads();
    let chunks: Vec<&mut [f64]> = mi_vec.chunks_mut(chunk_size).collect();
    // Iterate over all feature pairs in parallel.
    chunks
        .into_par_iter()
        .enumerate()
        .for_each(|(chunk_index, chunk)| {
            let start_index = chunk_index * chunk_size;
            let end_index = std::cmp::min(start_index + chunk_size, n_pairs);
            for i in start_index..end_index {
                let tmp_ind_f1 = feat_pairs[[i, 0]] as usize;
                let tmp_ind_f2 = feat_pairs[[i, 1]] as usize;
                chunk[i - start_index] = mi_optimal(
                    data.row(tmp_ind_f1).as_slice().unwrap(),
                    data.row(tmp_ind_f2).as_slice().unwrap(),
                    sliding_windows,
                    &features_sort_indices[tmp_ind_f1],
                    &features_sort_indices[tmp_ind_f2],
                );
            }
        });

    // Sort the vector of mutual information values in descending order. feat_pairs are also sorted.
    if sort_results {
        let (mi_vec_sorted, feat_pairs_sorted) = sort_mi_results(&mi_vec, &feat_pairs);
        mi_vec = mi_vec_sorted;
        feat_pairs = feat_pairs_sorted;
    }

    // Return the vector of mutual information values.
    (Array1::from(mi_vec), feat_pairs)
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
pub fn mi_optimal(
    f1: &[f64],
    f2: &[f64],
    sliding_windows: &Vec<Vec<usize>>,
    sort_ind_f1: &Vec<usize>,
    sort_ind_f2: &Vec<usize>,
) -> f64 {
    // Sort feature_1 and feature_2 in ascending order based on the sortperm of feature_1.
    let sorted_f1: Vec<f64> = sort_ind_f1.iter().map(|&i| f1[i]).collect();
    let sorted_f2: Vec<f64> = sort_ind_f1.iter().map(|&i| f2[i]).collect();

    // Initialize a vector to store mutual information values.
    let n_windows = sliding_windows.len();
    let mut mi_vec: Vec<f64> = vec![0.0; n_windows];

    // Iterate over all sliding windows.
    mi_vec.par_iter_mut().enumerate().for_each(|(i, mi)| {
        // Extract the sliding window.
        let window = &sliding_windows[i];
        let tmp_sta = window[0];
        let tmp_end = window[1];

        let tmp_sorted_f1 = &sorted_f1[tmp_sta..tmp_end];
        let tmp_sorted_f2 = &sorted_f2[tmp_sta..tmp_end];

        let tmp_sorted_ind_f1: Vec<usize> = (0..(tmp_end - tmp_sta)).map(|i| i).collect();
        let tmp_sorted_ind_f2: Vec<usize> = get_sort_indices_vecf64_slice(tmp_sorted_f2);

        // Calculate the mutual information.
        *mi = mi_fd(
            tmp_sorted_f1,
            tmp_sorted_f2,
            &tmp_sorted_ind_f1,
            &tmp_sorted_ind_f2,
            true,
        );
    });

    // Find the maximum mutual information value.
    // let sort_indices = get_sort_indices_vecf64(&mi_vec);
    // let mut mi_opt = mi_vec[sort_indices[mi_vec.len() - 1]];
    let sorted_mi_vec = sort_vec_f64(&mi_vec);
    let mut mi_opt = sorted_mi_vec[sorted_mi_vec.len() - 1];

    // Calculate the MI for the complete data.
    let mi_complete = mi_fd(f1, f2, sort_ind_f1, sort_ind_f2, true);

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
fn mi_fd(
    feat_1: &[f64],
    feat_2: &[f64],
    sort_ind_f1: &Vec<usize>,
    sort_ind_f2: &Vec<usize>,
    normalized: bool,
) -> f64 {
    // Generate the bins.
    let Ok((n_bins_f1, bin_width_f1, quantiles_f1)) = bins_fd(feat_1, sort_ind_f1) else {
        return 0.0;
    };
    let Ok((n_bins_f2, bin_width_f2, quantiles_f2)) = bins_fd(feat_2, sort_ind_f2) else {
        return 0.0;
    };

    // Calculate the mutual information.
    hist2mi(
        feat_1,
        feat_2,
        n_bins_f1,
        n_bins_f2,
        bin_width_f1,
        bin_width_f2,
        &quantiles_f1,
        &quantiles_f2,
        normalized,
    )
}

/// RectangularBinning (the adaptive partitioning approach): Freedman-Diaconis' rule (no assumption on the distribution)
/// + Returns the number of bins, the bin width, and the quantiles.
fn bins_fd(
    vec_x: &[f64],
    sort_indices: &Vec<usize>,
) -> Result<(usize, f64, Vec<f64>), Box<dyn Error>> {
    let len_vec = vec_x.len();
    let len_vec_f64 = len_vec as f64;
    let mut quantiles: Vec<f64> = vec![0.0; 4];

    quantiles[1] = vec_x[sort_indices[((len_vec_f64 * 0.5).round() as usize) - 1]];
    quantiles[2] = vec_x[sort_indices[((len_vec_f64 * 0.75).round() as usize) - 1]];
    quantiles[0] = vec_x[sort_indices[0]];
    quantiles[3] = vec_x[sort_indices[len_vec - 1]];

    let iqr = quantiles[2] - quantiles[1];
    let bin_width = 2.0 * iqr / (len_vec_f64).cbrt();
    let n_bins = ((quantiles[3] - quantiles[0]) / bin_width).ceil() as usize;

    if n_bins < 3 || n_bins > len_vec {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Invalid number of bins.",
        )));
    }

    Ok((n_bins, bin_width, quantiles))
}

/// Calculate the mutual information value between two vectors using the histogram method.
fn hist2mi(
    vec_1: &[f64],
    vec_2: &[f64],
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
