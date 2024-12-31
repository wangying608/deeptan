use chrono::Local;
use ndarray::prelude::*;
use ndarray_npy::NpzWriter;
use polars::prelude::*;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fs::File;
use std::path::Path;

mod mutualinfo;
mod processing;
mod slidingwindow;
mod sortf64;

use mutualinfo::iter_feat_pairs_mi;
use processing::{remove_feat_similar, rm_feat_low_cv};
use slidingwindow::init_windows_from_ratio;
use sortf64::get_sort_indices_vecf64;

/// Generate MIC relations between features with dynamic feature filtering for the next graph initialization.
///
/// **Steps**:
/// 1. Remove features with low coefficients of variation (using dynamic sliding windows).
/// 2. Detect similar features pairs [Optional] (using dynamic 2D sliding windows for maxmizing PCC(`abs=true`)) then remove redundant features.
/// 3. Compute MIC for each feature pair (using dynamic sliding windows).
/// 4. Filter out weak MIC values and corresponding feature pairs.
/// 5. Save sorted MIC values, feature pairs, processed input data, feature indices, similar feature pairs and input arguments.
///
/// **Input**:
/// + `path_output`: path to save the result files
/// + `data`: input data with shape (n_var, n_obs)
/// + `obs_names`: observation names
/// + `var_names`: variable names
/// + `check_sim`: whether to detect similar features pairs and remove redundant features
/// + `thre_cv`: threshold for removing features with low coefficients of variation
/// + `thre_pcc`: threshold for removing redundant features
/// + `thre_mi`: threshold for removing feature pairs with low mutual information
/// + `ratio_max_window`: maximum (window_size / num_samples)
/// + `ratio_min_window`: minimum (window_size / num_samples)
/// + `ratio_step_window`: window_size_step_len / num_samples
/// + `ratio_step_sliding`: sliding_step_len / num_samples
/// + `n_threads`: number of threads
///
/// **Output**:
/// + NPZ file
///     + `mi_values`: sorted mutual information values
///     + `feat_pairs`: corresponding feature pairs (edges)
///     + `processed_mat`: The processed input matrix after *Step 3*
///     + `mat_feat_indices`: The feature indices of the processed matrix
///     + `mat_simi_feat_pairs`: Similar feature pairs given by *Step 3*
///     + input_args: The input arguments
/// + Parquet file
///     + The named dataframe of `processed_mat`
///
pub fn mic_mat_with_data_filter(
    path_output: &str,
    data: &Array2<f64>,
    obs_names: &DataFrame,
    var_names: &Vec<String>,
    check_sim: bool,
    thre_cv: f64,
    thre_pcc: f64,
    thre_mi: f64,
    ratio_max_window: f64,
    ratio_min_window: f64,
    ratio_step_window: f64,
    ratio_step_sliding: f64,
    n_threads: usize,
) -> Result<(), Box<dyn Error>> {
    // Check available threads
    let mut num_threads = std::thread::available_parallelism().unwrap().get();
    if num_threads > 2 {
        num_threads -= 1;
    }
    if num_threads > n_threads && n_threads > 0 {
        num_threads = n_threads;
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();
    println!("\n⚡️  Using {} threads.\n", num_threads);

    // Initialize various sliding windows
    let sliding_windows = init_windows_from_ratio(
        data.ncols(),
        ratio_min_window,
        ratio_max_window,
        ratio_step_window,
        ratio_step_sliding,
    );

    // Pre-compute the sort indices for each feature.
    let features: Vec<Vec<f64>> = (0..data.nrows()).map(|i| data.row(i).to_vec()).collect();
    let mut features_sort_indices: Vec<Vec<usize>> = features
        .par_iter()
        .map(|feat| get_sort_indices_vecf64(feat))
        .collect();

    // Print start time
    println!("\nStart time: {:?}\n", Local::now());

    // 1. Remove low-CV features
    let (mut data_0, feat_indices_0, tmp_features_sort_indices) =
        rm_feat_low_cv(data, thre_cv, &sliding_windows, &features_sort_indices);
    features_sort_indices = tmp_features_sort_indices;
    println!(
        "Shape of data after removing features with low CV (coefficient of variation) values: {:?} (n_feat x n_obs)",
        data_0.shape()
    );

    // 2. Detect similar features pairs and remove redundant features
    let mut feat_indices_1 = feat_indices_0.clone();
    let mut simi_feat_pairs = Array2::<i64>::zeros((0, 2));

    if check_sim {
        println!("\nStart removing similar features.");
        (
            data_0,
            feat_indices_1,
            simi_feat_pairs,
            features_sort_indices,
        ) = remove_feat_similar(&data_0, thre_pcc, &sliding_windows, &features_sort_indices);
        println!(
            "Shape of data after removing similar features: {:?}",
            data_0.shape()
        );
    } else {
        println!("\nSkip removing similar features.");
    }

    // Print time
    let time_start = Local::now();
    println!("\nStart computing MIC relations: {:?}", time_start);

    // 3. Compute MIC for each feature pair
    let (mi_values, feat_pairs) =
        iter_feat_pairs_mi(&data_0, &sliding_windows, &features_sort_indices, true);

    // Print end time
    let time_end = Local::now();
    println!("End time:   {:?}", time_end);
    println!("Time cost:  {:?}\n", time_end - time_start);

    // 4. Remove low-MIC feature pairs.
    // mi_values and feat_pairs have been sorted. We check elements from the end.
    let last2keep = check_sorted_vals(&mi_values, thre_mi);
    // Keep pairs that idx >= last2keep
    let mi_values_o: Array1<f64> = mi_values.slice(s![..last2keep]).to_owned();
    let mut feat_pairs_o: Array2<i64> = feat_pairs.slice(s![..last2keep, ..]).to_owned();
    println!(
        "Number of feature pairs after removing weak MIC values: {}",
        last2keep + 1
    );

    // Get sorted unique features in pairs
    let flattened: Array1<i64> = feat_pairs_o.iter().copied().collect();
    let uniq_features: HashSet<i64> = flattened.into_iter().collect();
    let mut sorted_uniq_features: Vec<usize> = uniq_features.iter().map(|&x| x as usize).collect();
    sorted_uniq_features.sort();

    // Convert feat_pairs_o based on feat_indices_1
    let mut map_1: HashMap<usize, usize> = HashMap::new();
    sorted_uniq_features.iter().for_each(|&i| {
        map_1.insert(i, *feat_indices_1.get(i).unwrap());
    });
    feat_pairs_o.outer_iter_mut().for_each(|mut row| {
        row[0] = *map_1.get(&(row[0] as usize)).unwrap() as i64;
        row[1] = *map_1.get(&(row[1] as usize)).unwrap() as i64;
    });

    // Remove features of data_1 that are not in sorted_uniq_features
    data_0 = data_0.select(Axis(0), &sorted_uniq_features);

    // Convert features indices after MIC filtering
    sorted_uniq_features.par_iter_mut().for_each(|i| {
        *i = *map_1.get(&(*i)).unwrap();
    });

    // Convert feature indices to original indices.
    if check_sim {
        // Create a map from new indices (feat_indices_1) to original indices (feat_indices_0).
        let mut map_new2orig: HashMap<usize, usize> = HashMap::new();
        sorted_uniq_features.iter().for_each(|&i| {
            map_new2orig.insert(i, *feat_indices_0.get(i).unwrap());
        });
        // unwrap() is not safe because of the possibility of a None value.
        // if None occurs, the program will panic.

        // Update indices
        feat_pairs_o.outer_iter_mut().for_each(|mut row| {
            row[0] = *map_new2orig.get(&(row[0] as usize)).unwrap() as i64;
            row[1] = *map_new2orig.get(&(row[1] as usize)).unwrap() as i64;
        });
        sorted_uniq_features.par_iter_mut().for_each(|i| {
            *i = *map_new2orig.get(i).unwrap();
        });
        simi_feat_pairs.outer_iter_mut().for_each(|mut row| {
            row[0] = *map_new2orig.get(&(row[0] as usize)).unwrap() as i64;
            row[1] = *map_new2orig.get(&(row[1] as usize)).unwrap() as i64;
        });
    }

    let data_1_feat_indices_o: Array1<i64> =
        Array1::from_vec(sorted_uniq_features.par_iter().map(|&i| i as i64).collect());

    // 5. Save results
    // 5.1 Save results to a NPZ file
    save_npz(
        path_output,
        &mi_values_o,
        &feat_pairs_o,
        &data_0,
        &data_1_feat_indices_o,
        &simi_feat_pairs,
        thre_cv,
        thre_pcc,
        thre_mi,
        ratio_max_window,
        ratio_min_window,
        ratio_step_window,
        ratio_step_sliding,
    )?;
    // 5.2 Save processed matrix to a parquet file
    save_parquet(
        &format!("{}.parquet", path_output),
        &data_0,
        obs_names,
        var_names,
        &sorted_uniq_features,
    )?;

    Ok(())
}

/// Save all data to a NPZ file
fn save_npz(
    path_output_npz: &str,
    mi_values: &Array1<f64>,
    feat_pairs: &Array2<i64>,
    processed_mat: &Array2<f64>,
    feat_indices: &Array1<i64>,
    simi_feat_pairs: &Array2<i64>,
    thre_cv: f64,
    thre_pcc: f64,
    thre_mi: f64,
    ratio_max_window: f64,
    ratio_min_window: f64,
    ratio_step_window: f64,
    ratio_step_sliding: f64,
) -> Result<(), Box<dyn Error>> {
    let mut path_npz = format!("{}.npz", path_output_npz);
    // Check if path_output_npz ends with .npz
    if path_output_npz.ends_with(".npz") {
        path_npz = path_output_npz.to_string();
    }
    // Check if the file exists. If it does, add timestamp to the file name.
    if Path::new(&path_npz).exists() {
        let timestamp = Local::now().format("%Y%m%d%H%M%S");
        if path_output_npz.ends_with(".npz") {
            path_npz = path_output_npz.replace(".npz", &format!("_{}.npz", timestamp));
        } else {
            path_npz = format!("{}_{}.npz", path_output_npz, timestamp);
        }
    }
    let path_npz_c = &path_npz.clone();

    let file_npz = File::create(path_npz_c)?;
    let mut npz = NpzWriter::new_compressed(file_npz);
    npz.add_array("mi_values", mi_values)?;
    npz.add_array("feat_pairs", feat_pairs)?;
    npz.add_array("processed_mat", &processed_mat)?;
    npz.add_array("mat_feat_indices", &feat_indices)?;
    npz.add_array("mat_simi_feat_pairs", &simi_feat_pairs)?;
    // Save input parameters as length 1 ndarray
    npz.add_array("thre_cv", &Array1::from_vec(vec![thre_cv]))?;
    npz.add_array("thre_pcc", &Array1::from_vec(vec![thre_pcc]))?;
    npz.add_array("thre_mi", &Array1::from_vec(vec![thre_mi]))?;
    npz.add_array(
        "ratio_max_window",
        &Array1::from_vec(vec![ratio_max_window]),
    )?;
    npz.add_array(
        "ratio_min_window",
        &Array1::from_vec(vec![ratio_min_window]),
    )?;
    npz.add_array(
        "ratio_step_window",
        &Array1::from_vec(vec![ratio_step_window]),
    )?;
    npz.add_array(
        "ratio_step_sliding",
        &Array1::from_vec(vec![ratio_step_sliding]),
    )?;
    // Finish writing
    npz.finish()?;

    println!("Results have been saved to \"{}\"", path_npz_c);

    Ok(())
}

/// Save processed matrix to a parquet file
fn save_parquet(
    path_output_parquet: &str,
    processed_mat: &Array2<f64>,
    obs_names: &DataFrame,
    var_names: &Vec<String>,
    feat_indices: &Vec<usize>,
) -> Result<(), Box<dyn Error>> {
    // Check if path_output_parquet ends with .parquet
    let mut path_parquet = format!("{}.parquet", path_output_parquet);
    if path_output_parquet.ends_with(".parquet") {
        path_parquet = path_output_parquet.to_string();
    }
    // Check if the file exists. If it does, add timestamp to the file name.
    if Path::new(&path_parquet).exists() {
        let timestamp = Local::now().format("%Y%m%d%H%M%S");
        if path_output_parquet.ends_with(".parquet") {
            path_parquet =
                path_output_parquet.replace(".parquet", &format!("_{}.parquet", timestamp));
        } else {
            path_parquet = format!("{}_{}.parquet", path_output_parquet, timestamp);
        }
    }
    let path_parquet_c = &path_parquet.clone();

    // Pick saved var (feature) names based on feat_indices
    let saved_var_names = var_names
        .iter()
        .enumerate()
        .filter(|(i, _)| feat_indices.contains(i))
        .map(|(_, x)| x.clone())
        .collect::<Vec<_>>();

    // Create a new DataFrame with processed_mat, obs_names and the selected var_names
    let vec_columns: Vec<Column> = saved_var_names
        .iter()
        .zip(processed_mat.outer_iter())
        .map(|(name, values)| Column::new(name.into(), values.to_vec()))
        .collect();

    let mut df1 = obs_names.hstack(&vec_columns)?;

    let mut file = File::create(path_parquet_c)?;
    ParquetWriter::new(&mut file).finish(&mut df1)?;

    println!(
        "Processed matrix has been saved as a dataframe with obs names and feature names: \"{}\"",
        path_parquet_c
    );
    Ok(())
}

/// Read parquet file (n_obs x (1 + n_vars)) into ndarray (n_obs x n_vars)
pub fn read_parquet_to_array2d(
    path_parquet: &str,
) -> Result<(Array2<f64>, DataFrame, Vec<String>), Box<dyn Error>> {
    let mut file = std::fs::File::open(path_parquet).unwrap();
    let df = ParquetReader::new(&mut file).finish().unwrap();
    // let mat = df.to_ndarray::<f64>().unwrap();
    // let lf1 = LazyFrame::scan_parquet(path_parquet, Default::default())?;
    let obs_names = df.select(["obs_names"])?;
    let binding = df.drop("obs_names")?;
    let var_names_0 = binding.get_column_names();
    // Convert Vec<&PlSmallStr> to Vec<String>
    let var_names: Vec<String> = var_names_0.iter().map(|x| x.to_string()).collect();
    let mat = binding
        .to_ndarray::<Float64Type>(IndexOrder::Fortran)
        .unwrap()
        .t()
        .to_owned();
    Ok((mat, obs_names, var_names))
}

/// mi_values and feat_pairs have been sorted. We can check elements from the end.
fn check_sorted_vals(vals: &Array1<f64>, threshold: f64) -> usize {
    let num_pairs = vals.len();
    // Start from the end
    let mut chk_i = num_pairs - 1;
    while chk_i > 0 {
        // Stop if the current pair has a higher MI than the threshold
        if vals[chk_i] > threshold {
            break;
        }
        chk_i -= 1;
    }
    chk_i
}

/*
use ndarray_rand::rand;
use ndarray_npy::{read_npy, NpzReader};
use rand::Rng;

/// Random 2D ndarray generator
pub fn random_2d_array(n_feat: usize, n_samp: usize) -> Array2<f64> {
    let mut rng = rand::thread_rng();
    let data: Vec<f64> = (0..n_feat * n_samp).map(|_| rng.gen()).collect();
    Array2::from_shape_vec((n_feat, n_samp), data).unwrap()
}

/// Random 1D ndarray generator
pub fn random_two_simi_vectors(vlen: usize) -> (Array1<f64>, Array1<f64>) {
    // Generate two random 1d arrays with the same length <veclen>, containing f64 values
    let mut v1 = Array::<f64, _>::zeros(vlen);
    let mut v2 = Array::<f64, _>::zeros(vlen);
    let mut rng = rand::thread_rng();
    for x in v1.iter_mut() {
        *x = rng.gen();
    }
    for x in v2.iter_mut() {
        *x = rng.gen();
    }

    // Use v2 as noise, generate new v2 = v1 + 0.1 * v2
    let noise = 0.001;
    for (x, y) in v1.iter().zip(v2.iter_mut()) {
        *y = x + noise * *y;
    }

    // Min-max normalization
    v1 = Array1::from_vec(normalize_vecf64(&v1.to_vec()));
    v2 = Array1::from_vec(normalize_vecf64(&v2.to_vec()));

    println!("normed v1: {:?}", v1);
    println!("normed v2: {:?}", v2);
    (v1, v2)
}

/// Read NPZ file into ndarray (n_feat x n_samp)
pub fn read_npz_to_array2d(path_npz: &str) -> Result<Array2<f64>, Box<dyn Error>> {
    let mut npz = NpzReader::new(File::open(path_npz)?)?;
    let mat: Array2<f64> = npz.by_name("mat")?;
    Ok(mat)
}
pub fn read_npy_to_array2d(path_npy: &str) -> Result<Array2<f64>, Box<dyn Error>> {
    let mat: Array2<f64> = read_npy(path_npy)?;
    Ok(mat)
}

*/
