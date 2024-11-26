use crate::slidingwindow::init_windows_from_ratio;
use crate::sortf64::{get_sort_indices_vecf64, sort_vec_f64, sort_vecs_by_first};
use ndarray::prelude::*;
use rayon::prelude::*;

/// Min-max normalize a Vec<f64>.
pub fn normalize_vecf64(vec_x: &Vec<f64>) -> Vec<f64> {
    let sorted_vec = sort_vec_f64(vec_x);
    let min_x = sorted_vec[0];
    let max_x = sorted_vec[vec_x.len() - 1];

    let normalized_vec_x = vec_x
        .par_iter()
        .map(|&x| (x - min_x) / (max_x - min_x))
        .collect();

    normalized_vec_x
}

/// Normalize (min-max) a 2D array along the rows(features).
pub fn normalize_2d_array(array_2d: &Array2<f64>) -> Array2<f64> {
    let n_feat = array_2d.nrows();
    let n_samp = array_2d.ncols();

    let mut normalized_array_2d = Array2::zeros((n_feat, n_samp));

    // Normalization for each feature(row).
    array_2d.outer_iter().enumerate().for_each(|(i, row)| {
        let normalized_row_i = normalize_vecf64(&row.to_vec());
        normalized_array_2d
            .row_mut(i)
            .assign(&Array1::from(normalized_row_i));
    });

    normalized_array_2d
}

/// Calculate standard deviation of a 2D array for each row(feature).
/// Note:
/// + Features are rows, samples are columns. Features must be normalized before input.
pub fn std_dev_2d_array(array_2d: &Array2<f64>, sliding_windows: &Vec<Vec<usize>>) -> Vec<f64> {
    let n_feat = array_2d.nrows();
    // let n_samp = array_2d.ncols();
    let n_windows = sliding_windows.len();

    let mut std_dev_opt: Vec<f64> = vec![0.0; n_feat];

    // (Parallelized) Calculate the standard deviation for each feature, for each feature's sliding windows.
    std_dev_opt
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, std_dev)| {
            let sorted_feat_i = sort_vec_f64(&array_2d.row(i).to_vec());

            // Calculate the optimal standard deviation for each feature, through sliding windows.
            let mut std_dev_vec_tmp: Vec<f64> = vec![0.0; n_windows + 1];
            for (j, window) in sliding_windows.iter().enumerate() {
                let part_j = sorted_feat_i[window[0]..window[1]].to_vec();
                // Calculate standard deviation of the feature's part.
                std_dev_vec_tmp[j] = Array1::from_vec(part_j).std(1.0);
            }
            // Calculate standard deviation of the complete feature.
            std_dev_vec_tmp[n_windows] = Array1::from_vec(sorted_feat_i).std(1.0);

            // Check if None value exists.
            // if std_dev_vec_tmp.iter().any(|x| x.is_nan()) {
            //     panic!("NaN value detected in std_dev_vec_tmp.");
            // }

            // Substitude None with zero.
            std_dev_vec_tmp.iter_mut().for_each(|x| {
                if x.is_nan() {
                    *x = 0.0;
                }
            });
            // println!("std_dev_vec_tmp: {:?}", std_dev_vec_tmp);

            // Save the maximum standard deviation of the feature's sliding windows.
            *std_dev = *std_dev_vec_tmp
                .iter()
                .max_by(|x, y| x.partial_cmp(y).unwrap())
                .unwrap();
        });

    std_dev_opt
}

/// Remove features with low standard deviation (using dynamic sliding windows).
pub fn remove_feat_low_sd(
    array_2d: &Array2<f64>,
    thre_stddev: f64,
    ratio_max_window: f64,
    ratio_min_window: f64,
    ratio_step_window: f64,
    ratio_step_sliding: f64,
) -> (Array2<f64>, Vec<usize>) {
    // Init sliding windows
    let len_vec = array_2d.ncols();
    let sld_windows = init_windows_from_ratio(
        len_vec,
        ratio_min_window,
        ratio_max_window,
        ratio_step_window,
        ratio_step_sliding,
    );

    // Calculate standard deviation for each feature
    let sd_feats = std_dev_2d_array(array_2d, &sld_windows);

    // Filter features with low standard deviation
    let mut feat_idxs_filtered: Vec<usize> = Vec::new();
    for (i, sd) in sd_feats.iter().enumerate() {
        if *sd >= thre_stddev {
            feat_idxs_filtered.push(i);
        }
    }
    let data_ = array_2d.select(Axis(0), &feat_idxs_filtered);

    (data_, feat_idxs_filtered)
}

/// Check the similarity of two features (using dynamic 2D sliding windows for optimal PCC calculation).
/// + Accepets two vectors, sliding windows, and a threshold.
/// + Returns a bool (true if the features are similar, false otherwise).
pub fn check_similarity_2d(
    feat1: &Vec<f64>,
    feat2: &Vec<f64>,
    sliding_windows: &Vec<Vec<usize>>,
    thre_pcc: f64,
) -> bool {
    let sort_ind_f1 = get_sort_indices_vecf64(&feat1);
    let (sorted_feat1, sorted_feat2) = sort_vecs_by_first(&feat1, &feat2, &sort_ind_f1);
    let n_windows = sliding_windows.len();

    let mut pcc_vec: Vec<f64> = vec![0.0; n_windows];
    pcc_vec.par_iter_mut().enumerate().for_each(|(i, pcc)| {
        let window = &sliding_windows[i];
        let part1 = sorted_feat1[window[0]..window[1]].to_vec();
        let part2 = sorted_feat2[window[0]..window[1]].to_vec();
        *pcc = pearsoncc(&part1, &part2, true);
    });

    // Calculate PCC of the complete features.
    let pcc_complete = pearsoncc(&sorted_feat1, &sorted_feat2, true);

    // Find the maximum PCC of the features' sliding windows.
    let sorted_pcc_vec = sort_vec_f64(&pcc_vec);
    let mut pcc_opt = sorted_pcc_vec[sorted_pcc_vec.len() - 1];
    if pcc_opt < pcc_complete {
        pcc_opt = pcc_complete;
    }

    // Check if the maximum PCC is above the threshold.
    pcc_opt > thre_pcc
}

/// Detect similar features pairs (using dynamic 2D sliding windows for optimal PCC calculation) then remove redundant features.
pub fn remove_feat_similar(
    array_2d: &Array2<f64>,
    thre_pcc: f64,
    ratio_max_window: f64,
    ratio_min_window: f64,
    ratio_step_window: f64,
    ratio_step_sliding: f64,
) -> (Array2<f64>, Vec<usize>, Array2<i64>) {
    // Init sliding windows
    let len_vec = array_2d.ncols();
    let sld_windows = init_windows_from_ratio(
        len_vec,
        ratio_min_window,
        ratio_max_window,
        ratio_step_window,
        ratio_step_sliding,
    );

    // Calculate optimal PCC for each feature pair
    // Iter over all pairs of features (combinations of 2 rows)
    let mut similar_feat_pair_indices: Vec<Vec<usize>> = Vec::new();
    let mut feat_indices_to_remove: Vec<usize> = Vec::new();
    let mut feat_indices_to_keep: Vec<usize> = Vec::new();
    array_2d
        .axis_iter(Axis(0))
        .enumerate()
        .for_each(|(i, feat1)| {
            let feat1_vec = feat1.to_vec();
            array_2d
                .axis_iter(Axis(0))
                .enumerate()
                .for_each(|(j, feat2)| {
                    if i < j {
                        let feat2_vec = feat2.to_vec();
                        let similar_feat_pair =
                            check_similarity_2d(&feat1_vec, &feat2_vec, &sld_windows, thre_pcc);
                        if similar_feat_pair {
                            similar_feat_pair_indices.push(vec![i, j]);
                            feat_indices_to_remove.push(i);
                        }
                    }
                });
        });

    let mut array_new = array_2d.clone();

    if !feat_indices_to_remove.is_empty() {
        // Remove duplicated features from feat_indices_to_remove
        feat_indices_to_remove.sort_unstable();
        feat_indices_to_remove.dedup();

        // Remove features from array_2d
        // Init a 2D array f64 zeros with nrows = array_2d.nrows() - feat_indices_to_remove.len()
        // let mut array_new: Array2<f64> = Array2::zeros((array_2d.nrows() - feat_indices_to_remove.len(), array_2d.ncols()));
        array_new = Array2::zeros((
            array_2d.nrows() - feat_indices_to_remove.len(),
            array_2d.ncols(),
        ));
        array_2d
            .axis_iter(Axis(0))
            .enumerate()
            .for_each(|(i, feat)| {
                if !feat_indices_to_remove.contains(&i) {
                    // Copy the feature to the new array
                    feat_indices_to_keep.push(i);
                    array_new
                        .index_axis_mut(Axis(0), feat_indices_to_keep.len() - 1)
                        .assign(&feat);
                }
            });
    } else {
        // If there are no similar features, return the original array_2d
        feat_indices_to_keep = (0..array_2d.nrows()).collect();
    }

    //
    let mut nda_simi_feat_pairs: Array2<i64> = Array2::zeros((similar_feat_pair_indices.len(), 2));
    for (i, pair) in similar_feat_pair_indices.iter().enumerate() {
        for (j, &feat) in pair.iter().enumerate() {
            nda_simi_feat_pairs[[i, j]] = feat as i64;
        }
    }

    (array_new, feat_indices_to_keep, nda_simi_feat_pairs)
}

/// Pearson Correlation Coefficient
pub fn pearsoncc(vec1: &[f64], vec2: &[f64], abs: bool) -> f64 {
    // Check if vectors have the same length
    assert_eq!(vec1.len(), vec2.len());
    let len_v = vec1.len() as f64;
    let mean_v1 = vec1.iter().sum::<f64>() as f64 / len_v;
    let mean_v2 = vec2.iter().sum::<f64>() as f64 / len_v;
    let mut num = 0.0;
    let mut den_a = 0.0;
    let mut den_b = 0.0;
    for (&v1, &v2) in vec1.iter().zip(vec2.iter()) {
        num += (v1 - mean_v1) * (v2 - mean_v2);
        den_a += (v1 - mean_v1).powi(2);
        den_b += (v2 - mean_v2).powi(2);
    }
    let mut pcc = num / (den_a * den_b).sqrt();
    if abs {
        pcc = pcc.abs();
    }
    pcc
}
