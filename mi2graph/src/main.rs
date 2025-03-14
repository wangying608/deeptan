use clap::{Arg, ArgAction, Command};
use mi2graph::{mic_mat_with_data_filter, read_parquet_to_array2d};
use std::fs;
use std::path::Path;

fn main() {
    let matches = cli().get_matches();

    let path_in = matches.get_one::<String>("in").map(|s| s.as_str()).unwrap();
    let path_out = matches
        .get_one::<String>("out")
        .map(|s| s.as_str())
        .unwrap();
    let thre_cv: f64 = *matches
        .get_one("thre_cv")
        .expect("Error in reading threshold of coefficient of variation");
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
    let check_sim: bool = *matches
        .get_one("check_sim")
        .expect("Error in reading whether to check for similar features pairs");
    let n_threads: usize = *matches
        .get_one("nthreads")
        .expect("Error in reading number of threads");

    let (array_2d, obs_names, var_names) =
        read_parquet_to_array2d(path_in).expect("Failed to read file");

    if let Some(parent) = Path::new(path_out).parent() {
        if !parent.exists() {
            if let Err(e) = fs::create_dir_all(parent) {
                eprintln!("Failed to create directory for output: {}", e);
                std::process::exit(1);
            }
        }
    }

    mic_mat_with_data_filter(
        path_out,
        &array_2d,
        &obs_names,
        &var_names,
        check_sim,
        thre_cv,
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
        .version("0.2.0")
        .author("Chenhua Wu, chanhuawu@outlook.com")
        .about("Generate MIC relations between features with dynamic feature filtering for graph initialization.")
        .args([
            Arg::new("in")
                .short('i')
                .long("input")
                .help("Input a Parquet file path (a dataframe (`n_obs x (1 + n_vars)`), where the first column \"obs_names\" list sample IDs )")
                .required(true)
                .action(ArgAction::Set),
            Arg::new("out")
                .short('o')
                .long("output")
                .help("Output path")
                .required(true)
                .action(ArgAction::Set),
            Arg::new("thre_cv")
                .value_parser(clap::value_parser!(f64))
                .long("threcv")
                .help("Threshold of coefficient of variation for removing features")
                .default_value("0.1"),
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
                .default_value("0.98"),
            Arg::new("ratio_min_window")
                .value_parser(clap::value_parser!(f64))
                .long("minwin")
                .help("Minimum ratio of window size to the number of samples")
                .default_value("0.33"),
            Arg::new("ratio_step_window")
                .value_parser(clap::value_parser!(f64))
                .long("stepwin")
                .help("Step size of window size / number of samples")
                .default_value("0.07"),
            Arg::new("ratio_step_sliding")
                .value_parser(clap::value_parser!(f64))
                .long("stepsli")
                .help("Step size of sliding window / number of samples")
                .default_value("0.07"),
            Arg::new("check_sim")
                .short('s')
                .long("chksim")
                .help("Whether to detect similar features pairs and remove redundant features")
                .required(false)
                .action(ArgAction::SetTrue),
            Arg::new("nthreads")
                .value_parser(clap::value_parser!(usize))
                .long("threads")
                .short('t')
                .help("Number of threads (use all available threads - 1 by default)")
                .default_value("0"),
        ])
}
