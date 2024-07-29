use bincode;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use ndarray::prelude::*;
use noodles::vcf::variant::record::AlternateBases;
use noodles::{gff, vcf};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_pickle::{self, SerOptions};
use std::collections::HashMap;
use std::error::Error;
use std::io::{Read, Write};

/// Build a dictionary of gene information from a GFF file
pub fn build_gff_dict(path_gff: &str, path_output: &str) -> Result<(), Box<dyn Error>> {
    let mut reader = std::fs::File::open(path_gff)
        .map(std::io::BufReader::new)
        .map(gff::io::Reader::new)?;
    println!("\n🔍  Reading GFF file: {}\n... ...\n", path_gff);
    let mut gff_dict: HashMap<String, GeneInfo> = HashMap::new();
    for result in reader.records() {
        // If result is an error, skip the record
        // If result is an Ok, get the record
        let detect_ok = result.is_ok();
        if !detect_ok {
            continue;
        } else {
            let record = result.unwrap();

            if record.ty() == "gene" {
                let gene_id = record.attributes().get("ID").expect("gene_id not found");
                let gene_idx = gene_id.to_string();
                let pos_sta = record.start().get();
                let pos_end = record.end().get();
                let block_len = pos_end - pos_sta + 1;

                let gene_info = GeneInfo {
                    seqid: record.reference_sequence_name().to_string(),
                    start: pos_sta,
                    end: pos_end,
                    strand: record.strand().to_string(),
                    feature_type: record.ty().to_string(),
                    len: block_len,
                };

                gff_dict.insert(gene_idx, gene_info);
            }
        }
    }

    // Print number of blocks
    let n_blocks = gff_dict.len();
    if n_blocks == 0 {
        return Err("\n❗ No blocks found in GFF file".into());
    }
    println!(
        "\n😊  {} blocks of the genome have been collected.\n",
        n_blocks
    );

    // Save gff_dict to file
    let data2ser = GffDict { data: gff_dict };
    let mut path_o = String::from(path_output);
    if !path_output.ends_with(".bin.gz") {
        path_o = format!("{}.bin.gz", path_output);
    }
    let serialized = bincode::serialize(&data2ser)?;
    if let Err(e) = write_gz(&serialized, &path_o) {
        eprintln!("\n❗  Failed to save gff dict: {}\n", e);
    } else {
        println!("\n✅  GFF dict has been saved to: {}\n", path_o);
    }

    Ok(())
}

/// Encode genotypes from VCF file to a one-hot-index matrix.
pub fn vcf2encoded(
    path_vcf: &str,
    path_gff_dict: &str,
    path_output: &str,
) -> Result<(), Box<dyn Error>> {
    let vcf_dict = build_vcf_dict(path_vcf, path_gff_dict)?;

    let vcf_mat = encode_vcf(&vcf_dict);

    let b2g = block2gtype_for_s2g_sparse(&vcf_dict);

    // Write to file
    let path_o = write_encoded_vcf2pkl(
        &vcf_mat.mat,
        &vcf_mat.sample_ids,
        &vcf_mat.snp_ids,
        &vcf_dict.block_ids,
        &b2g,
        path_output,
    )?;
    println!("\n✅  Encoded VCF saved to 📄 {}\n", path_o);
    Ok(())
}

/// Write results to pickle file.
fn write_encoded_vcf2pkl(
    mat: &Vec<i8>,
    sample_ids: &Vec<String>,
    snp_ids: &Vec<String>,
    block_ids: &Vec<String>,
    block2gtype: &Vec<Vec<i64>>,
    path_output: &str,
) -> Result<String, Box<dyn Error>> {
    // Make sure path_output ends with .pkl
    let mut path_o = path_output.to_string();
    if path_o.ends_with(".pkl") {
        path_o.push_str(".gz");
    }
    if !path_o.ends_with(".pkl.gz") {
        path_o.push_str(".pkl.gz");
    }
    let path_pkl = &path_o.clone();

    // Transform data

    // Serialize data
    let serialized_sample_ids = serde_pickle::to_vec(sample_ids, SerOptions::default())?;
    let serialized_snp_ids = serde_pickle::to_vec(snp_ids, SerOptions::default())?;
    let serialized_mat = serde_pickle::to_vec(&mat, SerOptions::default())?;
    let serialized_block_ids = serde_pickle::to_vec(block_ids, SerOptions::default())?;
    let serialized_block2gtype = serde_pickle::to_vec(block2gtype, SerOptions::default())?;

    // let mut file_new = File::create(path_pkl)?;
    // file_new.write_all(&serialized_sample_ids)?;
    // file_new.write_all(&serialized_snp_ids)?;
    // file_new.write_all(&serialized_block_ids)?;
    // file_new.write_all(&serialized_block2gtype)?;
    // file_new.write_all(&serialized_mat)?;

    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&serialized_sample_ids)?;
    encoder.write_all(&serialized_snp_ids)?;
    encoder.write_all(&serialized_block_ids)?;
    encoder.write_all(&serialized_block2gtype)?;
    encoder.write_all(&serialized_mat)?;
    let compressed_data = encoder.finish()?;
    let mut file_new = std::fs::File::create(path_pkl)?;
    file_new.write_all(&compressed_data)?;

    Ok(path_pkl.to_owned())
}

fn build_vcf_dict(path_vcf: &str, path_gff_dict: &str) -> Result<VcfInfo, Box<dyn Error>> {
    // Read gff dict
    let decompressed_data = read_gz(path_gff_dict)?;
    let gff_dict_dec: GffDict = bincode::deserialize(&decompressed_data)?;
    let gff_dict = gff_dict_dec.data;

    let block_ids: Vec<String> = gff_dict.keys().map(|k| k.to_string()).collect();

    let mut reader_vcf = vcf::io::reader::Builder::default().build_from_path(path_vcf)?;
    let header = reader_vcf.read_header()?;
    let sample_ids = header
        .sample_names()
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<String>>();
    let n_samples = sample_ids.len();
    println!("\n🌾 Found {} samples in the VCF file.\n", n_samples);
    println!("🔎 Checking for SNPs in the VCF file...\n");

    let mut snp_dict: HashMap<String, SnpInfo> = HashMap::new();
    for result in reader_vcf.records() {
        if !result.is_ok() {
            continue;
        } else {
            let snp_check_kv = check_snp(&result.unwrap(), &gff_dict, &block_ids);
            if !snp_check_kv.snp_id.is_empty() {
                snp_dict.insert(snp_check_kv.snp_id, snp_check_kv.snp_info);
            }
        }
    }

    // // Read whole VCF file into memory
    // let records: Vec<vcf::Record> = reader_vcf.records().filter_map(|r| r.ok()).collect();
    // // Multi-threading:
    // let snp_dict_vec: Vec<SnpCheckKV> = records
    //     .into_iter()
    //     .filter_map(|r| check_snp(&r, &gff_dict, &block_ids).into())
    //     .collect();
    // let mut snp_dict: HashMap<String, SnpInfo> = HashMap::new();
    // for snp_check_kv in snp_dict_vec {
    //     if !snp_check_kv.snp_id.is_empty() {
    //         snp_dict.insert(snp_check_kv.snp_id, snp_check_kv.snp_info);
    //     }
    // }

    // Get SNP ids
    let mut snp_ids: Vec<String> = snp_dict.keys().map(|k| k.to_string()).collect();
    snp_ids.sort();

    // Get unique blocks' ids from snp_dict
    let mut blocks_ids: Vec<String> = snp_dict
        .values()
        .flat_map(|v| v.block_ids.clone())
        .collect();
    blocks_ids.sort();
    blocks_ids.dedup();

    println!(
        "\n✅  Found {} SNPs in {} genome blocks.\n",
        snp_ids.len(),
        blocks_ids.len()
    );

    let vcf_info = VcfInfo {
        snp_dict: snp_dict,
        snp_ids: snp_ids,
        sample_ids: sample_ids,
        block_ids: blocks_ids,
    };

    Ok(vcf_info)
}

fn check_snp(
    record_x: &vcf::Record,
    gff_dict: &HashMap<String, GeneInfo>,
    block_ids: &Vec<String>,
) -> SnpCheckKV {
    let mut snp_idx = String::new();
    let mut snp_info = SnpInfo {
        chrom: String::new(),
        pos: 0,
        ref_allele: String::new(),
        alt_alleles: Vec::new(),
        genotypes: Vec::new(),
        block_ids: Vec::new(),
    };

    let pos = record_x.variant_start();
    if pos.is_some() {
        let pos_x = pos.unwrap().unwrap().get();
        let seqid = record_x.reference_sequence_name().to_string();

        // Check position in gff dict
        let snp_found = check_pos(gff_dict, &block_ids, &seqid, pos_x);

        if snp_found.found {
            let ref_base = record_x.reference_bases().to_string();
            let alt_bases = record_x
                .alternate_bases()
                .iter()
                .map(|b| b.unwrap().to_string())
                .collect::<Vec<String>>();
            let genotypes = record_x
                .samples()
                .iter()
                .map(|s| s.as_ref().to_string())
                .collect::<Vec<String>>();
            snp_idx = format!("{}-{}", seqid, pos_x);
            snp_info = SnpInfo {
                chrom: seqid,
                pos: pos_x,
                ref_allele: ref_base,
                alt_alleles: alt_bases,
                genotypes: genotypes,
                block_ids: snp_found.block_ids,
            };
        }
    }
    return SnpCheckKV {
        snp_id: snp_idx,
        snp_info: snp_info,
    };
}

fn check_pos(
    gff_dict: &HashMap<String, GeneInfo>,
    block_ids: &Vec<String>,
    seq_name: &str,
    pos: usize,
) -> SnpFound {
    let mut snp_found = SnpFound {
        found: false,
        block_ids: vec![],
    };
    let n_genome_block = block_ids.len();
    let mut isfound: Vec<bool> = vec![false; n_genome_block];
    isfound.par_iter_mut().enumerate().for_each(|(i, isf)| {
        let gene_info = gff_dict.get(&block_ids[i]).unwrap();
        *isf = check_pos_single(gene_info, seq_name, pos);
    });
    // Check if any `true` exists in isfound
    if isfound.contains(&true) {
        let mut block_ids_found: Vec<String> = Vec::new();
        block_ids_found.extend(
            block_ids
                .iter()
                .zip(isfound)
                .filter(|(_, isf)| *isf)
                .map(|(b, _)| b.to_string()),
        );
        snp_found = SnpFound {
            found: true,
            block_ids: block_ids_found,
        };
    }
    return snp_found;
}

fn check_pos_single(gene_info: &GeneInfo, seq_name: &str, pos: usize) -> bool {
    if proc_seq_name(&gene_info.seqid) == proc_seq_name(seq_name) {
        if pos >= gene_info.start && pos <= gene_info.end {
            if gene_info.strand == "+" {
                return true;
            }
        }
    }
    false
}

/// For sparse layer initialization.
fn block2gtype_for_s2g_sparse(vcf_dict: &VcfInfo) -> Vec<Vec<i64>> {
    let n_blocks = vcf_dict.block_ids.len();
    let n_gt = vcf_dict.snp_ids.len();
    let mut block2gt: Vec<Vec<i64>> = vec![];
    for i in 0..n_blocks {
        let mut i_block_gt: Vec<i64> = vec![];
        for j in 0..n_gt {
            if vcf_dict
                .snp_dict
                .get(vcf_dict.snp_ids.get(j).unwrap())
                .unwrap()
                .block_ids
                .contains(vcf_dict.block_ids.get(i).unwrap())
            {
                i_block_gt.push(j as i64);
            }
        }
        block2gt.push(i_block_gt);
    }
    return block2gt;
}

fn encode_vcf(vcf_info: &VcfInfo) -> VcfMat {
    let sample_ids = &vcf_info.sample_ids;
    let snp_ids = &vcf_info.snp_ids;
    let n_snp = snp_ids.len();
    let n_sample = sample_ids.len();

    // Define dict
    let ten_combn: Vec<String> = vec![
        "A/A".to_string(),
        "C/C".to_string(),
        "G/G".to_string(),
        "T/T".to_string(),
        "A/C".to_string(),
        "A/G".to_string(),
        "A/T".to_string(),
        "C/G".to_string(),
        "C/T".to_string(),
        "G/T".to_string(),
    ];

    let mut encoded_mat: Array2<i8> = Array2::zeros((n_snp, n_sample));
    encoded_mat
        .axis_iter_mut(Axis(0))
        .enumerate()
        .for_each(|(i, mut row)| {
            let snp_x = vcf_info.snp_dict.get(&snp_ids[i]).unwrap();
            let encoded: Array1<i8> = Array1::from_vec(encode_snp_genotype(
                &snp_x.genotypes,
                &snp_x.ref_allele,
                &snp_x.alt_alleles,
                &ten_combn,
            ));
            row.assign(&encoded);
        });

    let mat_vec = encoded_mat.into_raw_vec();
    return VcfMat {
        mat: mat_vec,
        snp_ids: snp_ids.to_owned(),
        sample_ids: sample_ids.to_owned(),
    };
}

fn encode_snp_genotype(
    genotypes: &Vec<String>,
    ref_allele: &str,
    alt_alleles: &Vec<String>,
    base_combn: &Vec<String>,
) -> Vec<i8> {
    // Init encoded_genotypes: Vec<u8> with length genotypes.len()
    let mut encoded_genotypes: Vec<i8> = vec![0; genotypes.len()];
    encoded_genotypes
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, x)| {
            *x = encode_snp_gt(
                genotypes.get(i).unwrap(),
                ref_allele,
                alt_alleles,
                base_combn,
            );
        });
    return encoded_genotypes;
}

fn encode_snp_gt(
    gt: &String,
    ref_allele: &str,
    alt_alleles: &Vec<String>,
    base_combn: &Vec<String>,
) -> i8 {
    let hot_code: i8 = 0;

    // Define the map of GT -> base
    let mut gtype_map: HashMap<String, String> = HashMap::new();
    gtype_map.insert(".".to_string(), "N".to_string());
    gtype_map.insert("0".to_string(), ref_allele.to_string());
    for alt_x in 0..alt_alleles.len() {
        gtype_map.insert((alt_x + 1).to_string(), alt_alleles[alt_x].to_string());
    }

    //
    let gt_x = gt
        .split(":")
        .collect::<Vec<&str>>()
        .get(0)
        .to_owned()
        .unwrap()
        .to_string();

    let mut gt_split: Vec<&str> = vec![];
    if gt.contains("/") {
        gt_split = gt_x.split("/").collect();
    } else {
        gt_split = gt_x.split("|").collect();
    }

    let a_1 = gtype_map
        .get(&gt_split.get(0).unwrap().to_string())
        .unwrap();
    let a_2 = gtype_map
        .get(&gt_split.get(1).unwrap().to_string())
        .unwrap();
    let mut allele_sorted = "   ".to_string();
    if a_1 > a_2 {
        allele_sorted = format!("{}/{}", a_2, a_1);
    } else {
        allele_sorted = format!("{}/{}", a_1, a_2);
    }

    // Find allele_sorted in ten_combn
    let idx = base_combn.iter().position(|x| x == &allele_sorted);
    if idx.is_none() {
        return hot_code;
    } else {
        let which_hot = (idx.unwrap() + 1) as i8;
        return which_hot;
    }
}

fn write_gz(data: &Vec<u8>, path_output: &str) -> Result<(), Box<dyn Error>> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::best());
    encoder.write_all(&data)?;
    let compressed_data = encoder.finish()?;

    let mut file_o = std::fs::File::create(path_output)?;
    file_o.write_all(&compressed_data)?;
    Ok(())
}

fn read_gz(path_gz: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    let file = std::fs::File::open(path_gz)?;
    let mut decoder = GzDecoder::new(file);
    let mut data = Vec::new();
    decoder.read_to_end(&mut data)?;
    Ok(data)
}

fn proc_seq_name(seq_name: &str) -> String {
    let seq_name_lowcase = seq_name.to_lowercase();
    if seq_name_lowcase.contains("chr") {
        return seq_name_lowcase.replace("chr", "");
    } else {
        return seq_name_lowcase;
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct SnpInfo {
    chrom: String,
    pos: usize,
    ref_allele: String,
    alt_alleles: Vec<String>,
    genotypes: Vec<String>,
    block_ids: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeneInfo {
    seqid: String,
    start: usize,
    end: usize,
    strand: String,
    feature_type: String,
    len: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GffDict {
    data: HashMap<String, GeneInfo>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SnpFound {
    found: bool,
    block_ids: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct VcfInfo {
    snp_dict: HashMap<String, SnpInfo>,
    snp_ids: Vec<String>,
    sample_ids: Vec<String>,
    block_ids: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct VcfMat {
    mat: Vec<i8>,
    snp_ids: Vec<String>,
    sample_ids: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SnpCheckKV {
    snp_id: String,
    snp_info: SnpInfo,
}
