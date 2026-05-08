use rayon::slice::ParallelSliceMut;

/// Sort two Vec<f64> in ascending order based on the sortperm of the first vector.
pub fn sort_vecs_by_first(
    vec_1: &Vec<f64>,
    vec_2: &Vec<f64>,
    sort_ind_f1: &Vec<usize>,
) -> (Vec<f64>, Vec<f64>) {
    // let sort_ind_f1 = get_sort_indices_vecf64(vec_1);

    // Sort the vectors based on the sorted indices
    let sorted_vec_1 = sort_ind_f1.iter().map(|&i| vec_1[i]).collect();
    let sorted_vec_2 = sort_ind_f1.iter().map(|&i| vec_2[i]).collect();

    // return sorted vectors
    (sorted_vec_1, sorted_vec_2)
}

/// Sort Vec<f64>
pub fn sort_vec_f64(vec_f64: &Vec<f64>) -> Vec<f64> {
    let mut vc = vec_f64.clone();
    vc.sort_unstable_by(|a, b| a.total_cmp(b));
    vc
}

/// Get sort indices of a Vec<f64>.
pub fn get_sort_indices_vecf64(vec_x: &Vec<f64>) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..vec_x.len()).collect();
    // Sort the indices based on the values in vec_x
    // indices.sort_unstable_by(|&i, &j| vec_x[i].partial_cmp(&vec_x[j]).unwrap());
    indices.par_sort_unstable_by(|&i, &j| vec_x[i].partial_cmp(&vec_x[j]).unwrap());
    indices
}
pub fn get_sort_indices_vecf64_slice(vec_x: &[f64]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..vec_x.len()).collect();
    indices.par_sort_unstable_by(|&i, &j| vec_x[i].partial_cmp(&vec_x[j]).unwrap());
    indices
}
