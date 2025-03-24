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

pub fn rm_feat_low_cv(
    array_2d: &Array2<f64>,
    thre_cv: f64,
    sliding_windows: &Vec<Vec<usize>>,
    feat_sort_indices: &Vec<Vec<usize>>,
) -> (Array2<f64>, Vec<usize>, Vec<Vec<usize>>) {
    // Calculate coefficient of variation for each feature
    let cv_vals = cv_array2d(array_2d, sliding_windows, feat_sort_indices);

    // Filter features with low CV
    let mut feat_idxs_saved: Vec<usize> = Vec::new();
    for (i, cv) in cv_vals.iter().enumerate() {
        if *cv >= thre_cv {
            feat_idxs_saved.push(i);
        }
    }
    let data_ = array_2d.select(Axis(0), &feat_idxs_saved);
    let feat_sort_indices_: Vec<Vec<usize>> = feat_idxs_saved
        .iter()
        .map(|&i| feat_sort_indices[i].clone())
        .collect();
    (data_, feat_idxs_saved, feat_sort_indices_)
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

/*
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
fn stddev_array2d(array_2d: &Array2<f64>, sliding_windows: &Vec<Vec<usize>>) -> Vec<f64> {
    let n_feat = array_2d.nrows();
    let n_windows = sliding_windows.len();
    let mut stddev_opt: Vec<f64> = vec![0.0; n_feat];

    // (Parallelized) Calculate the standard deviation for each feature, for each feature's sliding windows.
    stddev_opt
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, std_dev)| {
            let sorted_feat_i = sort_vec_f64(&array_2d.row(i).to_vec());

            // Calculate the optimal standard deviation for each feature, through sliding windows.
            let mut tmp_stddev_vals: Vec<f64> = vec![0.0; n_windows + 1];
            for (j, window) in sliding_windows.iter().enumerate() {
                let part_j = sorted_feat_i[window[0]..window[1]].to_vec();
                // Calculate standard deviation of the feature's part.
                tmp_stddev_vals[j] = Array1::from_vec(part_j).std(1.0);
            }
            // Calculate standard deviation of the complete feature.
            tmp_stddev_vals[n_windows] = Array1::from_vec(sorted_feat_i).std(1.0);

            // Check if None value exists.
            // if tmp_stddev_vals.iter().any(|x| x.is_nan()) {
            //     panic!("NaN value detected in tmp_stddev_vals.");
            // }

            // Substitude None with zero.
            tmp_stddev_vals.iter_mut().for_each(|x| {
                if x.is_nan() {
                    *x = 0.0;
                }
            });
            // println!("tmp_stddev_vals: {:?}", tmp_stddev_vals);

            // Save the maximum standard deviation of the feature's sliding windows.
            *std_dev = *tmp_stddev_vals
                .iter()
                .max_by(|x, y| x.partial_cmp(y).unwrap())
                .unwrap();
        });

    stddev_opt
}

/// Remove features with low standard deviation (using dynamic sliding windows).
pub fn remove_feat_low_sd(
    array_2d: &Array2<f64>,
    thre_stddev: f64,
    sliding_windows: &Vec<Vec<usize>>,
) -> (Array2<f64>, Vec<usize>) {
    // Calculate standard deviation for each feature
    let sd_vals = stddev_array2d(array_2d, sliding_windows);

    // Filter features with low standard deviation
    let mut feat_idxs_saved: Vec<usize> = Vec::new();
    for (i, sd) in sd_vals.iter().enumerate() {
        if *sd >= thre_stddev {
            feat_idxs_saved.push(i);
        }
    }
    let data_ = array_2d.select(Axis(0), &feat_idxs_saved);

    (data_, feat_idxs_saved)
}
*/
