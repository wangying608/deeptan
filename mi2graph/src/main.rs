use clap::{Arg, ArgAction, Command};
use mi2graph::{mi_mat_with_data_filter, read_npy_to_array2d};

fn main() {
    let matches = cli().get_matches();

    let path_in = matches.get_one::<String>("in").map(|s| s.as_str()).unwrap();
    let path_out = matches
        .get_one::<String>("out")
        .map(|s| s.as_str())
        .unwrap();
    let thre_sd: f64 = *matches
        .get_one("thre_sd")
        .expect("Error in reading threshold of standard deviation");
    let thre_pcc: f64 = *matches
        .get_one("thre_pcc")
        .expect("Error in reading threshold of PCC");
    let thre_mi: f64 = *matches
        .get_one("thre_mi")
        .expect("Error in reading threshold of mutual information");
    let ratio_max_window: f64 = *matches
        .get_one("ratio_max_window")
        .expect("Error in reading maximum ratio of window size to the number of samples");
    let ratio_min_window: f64 = *matches
        .get_one("ratio_min_window")
        .expect("Error in reading minimum ratio of window size to the number of samples");
    let ratio_step_window: f64 = *matches
        .get_one("ratio_step_window")
        .expect("Error in reading step size of window size / number of samples");
    let ratio_step_sliding: f64 = *matches
        .get_one("ratio_step_sliding")
        .expect("Error in reading step size of sliding window / number of samples");
    let n_threads: usize = *matches
        .get_one("nthreads")
        .expect("Error in reading number of threads");

    // let rand_2d_array = read_npz_to_array2d(path_in).expect("Failed to read file");
    let rand_2d_array = read_npy_to_array2d(path_in).expect("Failed to read file");

    mi_mat_with_data_filter(
        path_out,
        &rand_2d_array,
        thre_sd,
        thre_pcc,
        thre_mi,
        ratio_max_window,
        ratio_min_window,
        ratio_step_window,
        ratio_step_sliding,
        n_threads,
    )
    .expect("Failed to read file");
}

/// Define the CLI interface
fn cli() -> Command {
    Command::new("mi2graph")
        .version("0.1.0")
        .author("Chenhua Wu, chanhuawu@outlook.com")
        .about("A Rust implementation of generating a mutual information matrix with dynamic data filtering for the graph initialization.")
        .args([
            Arg::new("in")
                .short('i')
                .long("input")
                .help("Input NumPy NPY file path (A matrix with a shape of n_feat x n_samp)")
                .required(true)
                .action(ArgAction::Set),
            Arg::new("out")
                .short('o')
                .long("output")
                .help("Output Numpy NPZ file path")
                .required(true)
                .action(ArgAction::Set),
            Arg::new("thre_sd")
                .value_parser(clap::value_parser!(f64))
                .long("thresd")
                .help("Threshold of standard deviation for removing features")
                .default_value("0.01"),
            Arg::new("thre_pcc")
                .value_parser(clap::value_parser!(f64))
                .long("threpcc")
                .help("Threshold of PCC for removing redundant features")
                .default_value("0.95"),
            Arg::new("thre_mi")
                .value_parser(clap::value_parser!(f64))
                .long("thremi")
                .help("Threshold of mutual information for removing edges (feature pairs)")
                .default_value("0.05"),
            Arg::new("ratio_max_window")
                .value_parser(clap::value_parser!(f64))
                .long("maxwin")
                .help("Maximum ratio of window size to the number of samples")
                .default_value("0.99"),
            Arg::new("ratio_min_window")
                .value_parser(clap::value_parser!(f64))
                .long("minwin")
                .help("Minimum ratio of window size to the number of samples")
                .default_value("0.8"),
            Arg::new("ratio_step_window")
                .value_parser(clap::value_parser!(f64))
                .long("stepwin")
                .help("Step size of window size / number of samples")
                .default_value("0.08"),
            Arg::new("ratio_step_sliding")
                .value_parser(clap::value_parser!(f64))
                .long("stepsli")
                .help("Step size of sliding window / number of samples")
                .default_value("0.05"),
            Arg::new("nthreads")
                .value_parser(clap::value_parser!(usize))
                .long("threads")
                .short('t')
                .help("Number of threads [default: 0, use all available threads]")
                .default_value("0"),
        ])
}
