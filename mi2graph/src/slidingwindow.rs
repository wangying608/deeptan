/// Initialize windows
/// + `step_win_size` and `step_slide` are the step widths of window size and slide, respectively.
/// + The minimum step width is `1` which means sliding only one data point.
fn init_windows(
    len_vec: usize,
    min_win_size: usize,
    max_win_size: usize,
    step_win_size: usize,
    step_slide: usize,
) -> Vec<Vec<usize>> {
    // Check the validity of the parameters.
    if min_win_size < 2 {
        panic!("Minimum window size must be greater than or equal to 2.");
        // If the minimum window size is 2, the step width of window size must be 1.
    }
    if max_win_size < min_win_size || max_win_size > len_vec {
        panic!("Maximum window size must be greater than or equal to the minimum window size and less than or equal to the length of the vector.");
    }
    if step_win_size < 1 || step_slide < 1 {
        panic!("Step widths of window size and slide must be greater than or equal to 1.");
    }

    // Initialize the windows
    let mut windows: Vec<Vec<usize>> = Vec::new();

    let win_sizes = (min_win_size..=max_win_size).step_by(step_win_size);

    // Loop through the window sizes
    for win_size in win_sizes {
        // Calculate the number of windows
        let n_windows = (len_vec - win_size).div_ceil(step_slide);

        // Loop through the windows
        for _w in 0..n_windows {
            let start = _w * step_slide;
            let end = start + win_size;
            if end > (len_vec - 1) {
                break;
                // Raise an error
                // panic!("Window size is larger than the length of the vector.");
            }
            if start >= end {
                // Raise an error
                panic!("Window size is larger than the length of the vector.");
            }
            windows.push(vec![start, end]);
        }
    }
    windows
}

/// Init sliding windows from ratio
pub fn init_windows_from_ratio(
    len_vec: usize,
    ratio_min_window: f64,
    ratio_max_window: f64,
    ratio_step_window: f64,
    ratio_step_sliding: f64,
) -> Vec<Vec<usize>> {
    // Check input
    assert!(ratio_min_window > 0.0);
    assert!(ratio_max_window < 1.0);
    assert!(ratio_min_window < ratio_max_window);
    assert!(ratio_step_window > 0.0);
    assert!(ratio_step_sliding > 0.0);
    assert!(len_vec > 10);

    // Init sliding windows
    let min_win_size = (len_vec as f64 * ratio_min_window).round() as usize;
    let max_win_size = (len_vec as f64 * ratio_max_window).round() as usize;
    let step_win_size = (len_vec as f64 * ratio_step_window).round() as usize;
    let step_slide = (len_vec as f64 * ratio_step_sliding).round() as usize;
    let sld_windows = init_windows(
        len_vec,
        min_win_size,
        max_win_size,
        step_win_size,
        step_slide,
    );
    sld_windows
}
