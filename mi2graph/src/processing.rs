use crate::sortf64::{sort_vec_f64, sort_vecs_by_first};
use ndarray::prelude::*;
use rayon::prelude::*;

fn cv_array2d(
    array_2d: &Array2<f64>,
    sliding_windows: &Vec<Vec<usize>>,
    feat_sort_indices: &Vec<Vec<usize>>,
) -> Vec<f64> {
    let n_feat = array_2d.nrows();
    let n_windows = sliding_windows.len();
    let mut cv_opt: Vec<f64> = vec![0.0; n_feat];

    // Compute CV for each feature, for each feature's sliding windows.
    cv_opt.par_iter_mut().enumerate().for_each(|(i, cv)| {
        let f_i = array_2d.row(i).to_vec();
        let sorted_f_i = feat_sort_indices[i]
            .iter()
            .map(|&i| f_i[i])
            .collect::<Vec<f64>>();

        // Compute the optimal CV for each feature through sliding windows.
        let mut tmp_cv_vals: Vec<f64> = vec![0.0; n_windows + 1];
        for (j, window) in sliding_windows.iter().enumerate() {
            let part_j = sorted_f_i[window[0]..window[1]].to_vec();

            // CV of the feature's part.
            let tmp_array = Array1::from(part_j);
            let part_j_sd = &tmp_array.std(1.0);
            let part_j_mean = &tmp_array.mean().unwrap().abs();
            if *part_j_mean < 0.0001 {
                tmp_cv_vals[j] = 0.0;
            } else {
                tmp_cv_vals[j] = part_j_sd / part_j_mean;
            }
        }

        // Compute CV of the complete feature.
        let tmp_array = Array1::from(sorted_f_i);
        let complete_feat_sd = &tmp_array.std(1.0);
        let complete_feat_mean = &tmp_array.mean().unwrap().abs();
        if *complete_feat_mean < 0.01 {
            tmp_cv_vals[n_windows] = 0.0;
        } else {
            tmp_cv_vals[n_windows] = complete_feat_sd / complete_feat_mean;
        }

        *cv = *tmp_cv_vals
            .iter()
            .max_by(|x, y| x.partial_cmp(y).unwrap())
            .unwrap();
    });

    cv_opt
}

/// Filters features based on coefficient of variation (CV) criteria.
///
/// Args:
///     array_2d: 2D array of feature data (features × samples)
///     cv_threshold: Minimum CV value to retain features (ignored if n_features_to_select > 0)
///     n_features_to_select: Number of top-CV features to retain (0 = use cv_threshold instead)
///     sliding_windows: Window indices for CV calculation
///     feat_sort_indices: Precomputed sorted feature indices for each window
///
/// Returns:
///     Tuple of (filtered_data, kept_feat_indices, filtered_feat_sort_indices) where:
///     - filtered_data: Subset of input array with selected features
///     - kept_feat_indices: Original indices of retained features
///     - filtered_feat_sort_indices: Subset of feat_sort_indices for retained features
pub fn rm_feat_low_cv(
    array_2d: &Array2<f64>,
    cv_threshold: f64,
    n_features_to_select: usize,
    sliding_windows: &Vec<Vec<usize>>,
    feat_sort_indices: &Vec<Vec<usize>>,
) -> (Array2<f64>, Vec<usize>, Vec<Vec<usize>>) {
    debug_assert_eq!(
        array_2d.len_of(Axis(0)),
        feat_sort_indices.len(),
        "Input dimension mismatch: number of features in array_2d and feat_sort_indices must be equal."
    );

    // Calculate coefficient of variation for each feature
    let cv_vals = cv_array2d(array_2d, sliding_windows, feat_sort_indices);

    debug_assert_eq!(
        array_2d.len_of(Axis(0)),
        cv_vals.len(),
        "CV calculation returned a vector of unexpected length."
    );

    // Filter features
    let kept_feat_indices: Vec<usize> = if n_features_to_select > 0 {
        let mut cv_with_indices: Vec<(usize, &f64)> = cv_vals.iter().enumerate().collect();

        // Sort by CV in descending order
        cv_with_indices.sort_unstable_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        let mut indices: Vec<usize> = cv_with_indices
            .into_iter()
            .take(n_features_to_select)
            .map(|(i, _)| i)
            .collect();
        // Sort indices for predictable selection order
        indices.sort_unstable();
        indices
    } else {
        // Original behavior: filter features with low CV
        cv_vals
            .iter()
            .enumerate()
            .filter(|(_i, &cv)| cv >= cv_threshold)
            .map(|(i, _cv)| i)
            .collect()
    };

    let filtered_data = array_2d.select(Axis(0), &kept_feat_indices);
    let filtered_feat_sort_indices: Vec<Vec<usize>> = kept_feat_indices
        .iter()
        .map(|&i| feat_sort_indices[i].clone())
        .collect();
    (filtered_data, kept_feat_indices, filtered_feat_sort_indices)
}

/// Check the similarity of two features (using dynamic 2D sliding windows for optimal PCC calculation).
/// + Accepets two vectors, sliding windows, and a threshold.
/// + Returns a bool (true if the features are similar, false otherwise).
fn check_similarity_2d(
    feat1: &Vec<f64>,
    feat2: &Vec<f64>,
    sliding_windows: &Vec<Vec<usize>>,
    thre_pcc: f64,
    sort_ind_f1: &Vec<usize>,
) -> bool {
    let (sorted_feat1, sorted_feat2) = sort_vecs_by_first(&feat1, &feat2, sort_ind_f1);
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
    sliding_windows: &Vec<Vec<usize>>,
    feat_sort_indices: &Vec<Vec<usize>>,
) -> (Array2<f64>, Vec<usize>, Array2<i64>, Vec<Vec<usize>>) {
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
                        let similar_feat_pair = check_similarity_2d(
                            &feat1_vec,
                            &feat2_vec,
                            sliding_windows,
                            thre_pcc,
                            &feat_sort_indices[i],
                        );
                        if similar_feat_pair {
                            similar_feat_pair_indices.push(vec![i, j]);
                            feat_indices_to_remove.push(i);
                        }
                    }
                });
        });

    let mut array_new = array_2d.clone();
    let mut feat_sort_indices_new: Vec<Vec<usize>> = feat_sort_indices.clone();

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

        feat_sort_indices_new = feat_indices_to_keep
            .iter()
            .map(|&i| feat_sort_indices[i].clone())
            .collect();
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

    (
        array_new,
        feat_indices_to_keep,
        nda_simi_feat_pairs,
        feat_sort_indices_new,
    )
}

/// Pearson Correlation Coefficient
fn pearsoncc(vec1: &[f64], vec2: &[f64], abs: bool) -> f64 {
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
