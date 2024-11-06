use std::fs::File;
use std::io::Write;
use tempfile::{tempdir, TempDir};
use ndarray::Array2;
use std::str::FromStr;
use bitvec::vec::BitVec;
use tensorflow::{DataType, Graph, ops, SavedModelBundle, Scope, Session, SessionOptions, SessionRunArgs, Tensor};

const SAVED_MODEL_PB: &[u8] = include_bytes!("../assets/vae_encoder/saved_model.pb");
const VARIABLES_INDEX: &[u8] = include_bytes!("../assets/vae_encoder/variables/variables.index");
const VARIABLES_DATA: &[u8] = include_bytes!("../assets/vae_encoder/variables/variables.data-00000-of-00001");
const CENTROID_STR: &str = include_str!("../assets/lf_kmeans_10k_centroids_20241025.csv");

lazy_static::lazy_static! {
    pub static ref ENCODER_MODEL: EncoderModel = build_encoder_model();
}

pub struct EncoderModel {
    pub encoder: SavedModelBundle,
    pub graph: Graph,
    pub centroids: Tensor<f32>,
}

impl EncoderModel {
    pub fn transform(&self, input_data: &BitVec) -> Vec<i32> {
        let lf_array = encode(&self, input_data);
        assign_cluster_labels(&self.centroids, &lf_array)
    }
}

fn build_encoder_model() -> EncoderModel {
    let (encoder, graph) = load_encoder_model();
    let centroids = load_cluster_centroids();

    EncoderModel {
        encoder,
        graph,
        centroids
    }
}

fn load_cluster_centroids() -> Tensor<f32> {
    let centroid_vec = CENTROID_STR
        .lines()
        .map(|line| {
            line.split(',')
                .map(|value| f32::from_str(value.trim()).unwrap())
                .collect()
        })
        .collect::<Vec<Vec<f32>>>();

    let array: Array2<f32> = Array2::from_shape_vec((centroid_vec.len(), centroid_vec[0].len()), centroid_vec.concat()).unwrap();
    let tensor = Tensor::new(&[array.shape()[0] as u64, array.shape()[1] as u64])
        .with_values(array.as_slice().unwrap()).unwrap();

    tensor
}

fn load_encoder_model() -> (SavedModelBundle, Graph) {
    let session_options = SessionOptions::new();
    let mut graph = Graph::new();
    let temp_dir = model_bytes_to_tempdir();
    let saved_model = SavedModelBundle::load(&session_options, vec!["serve"], &mut graph, temp_dir.path().to_str().unwrap()).unwrap();

    (saved_model, graph)
}

fn model_bytes_to_tempdir() -> TempDir {
    let temp_dir = tempdir().unwrap();
    let model_path = temp_dir.path().join("saved_model.pb");
    let mut saved_model_file = File::create(&model_path).unwrap();
    saved_model_file.write_all(SAVED_MODEL_PB).unwrap();

    let variables_dir = temp_dir.path().join("variables");
    std::fs::create_dir_all(&variables_dir).unwrap();
    let variables_index_path = variables_dir.join("variables.index");
    let variables_data_path = variables_dir.join("variables.data-00000-of-00001");

    let mut variables_index_file = File::create(&variables_index_path).unwrap();
    variables_index_file.write_all(VARIABLES_INDEX).unwrap();

    let mut variables_data_file = File::create(&variables_data_path).unwrap();
    variables_data_file.write_all(VARIABLES_DATA).unwrap();

    temp_dir
}

fn encode(model: &EncoderModel, input_data: &BitVec) -> Tensor<f32> {
    let input_data = input_data.iter().map(|b| if *b {1} else {0}).collect::<Vec<i64>>();
    let input_tensor = Tensor::new(&[1, input_data.len() as u64]).with_values(&input_data).unwrap();

    let input_operation = model.graph.operation_by_name("serving_default_dense_input").unwrap().unwrap();
    let output_operation = model.graph.operation_by_name("StatefulPartitionedCall").unwrap().unwrap();

    let mut run_args = SessionRunArgs::new();
    run_args.add_feed(&input_operation, 0, &input_tensor);

    let output_token = run_args.request_fetch(&output_operation, 0);
    model.encoder.session.run(&mut run_args).unwrap();

    run_args.fetch(output_token).unwrap()
}

pub fn assign_cluster_labels(centroids: &Tensor<f32>, lf_array: &Tensor<f32>) -> Vec<i32> {
    let mut scope = Scope::new_root_scope();
    let mut run_args = SessionRunArgs::new();

    let centroids_input = ops::Placeholder::new()
        .dtype(DataType::Float)
        .shape(centroids.dims())
        .build(&mut scope)
        .unwrap();

    let lf_input = ops::Placeholder::new()
        .dtype(DataType::Float)
        .shape(lf_array.dims())
        .build(&mut scope)
        .unwrap();

    run_args.add_feed(&centroids_input, 0, &centroids);
    run_args.add_feed(&lf_input, 0, &lf_array);

    let begin_tensor = ops::Const::new()
        .dtype(DataType::Int32)
        .value(Tensor::new(&[2]).with_values(&[0, 0]).unwrap())
        .build(&mut scope)
        .unwrap();

    let size_tensor = ops::Const::new()
        .dtype(DataType::Int32)
        .value(Tensor::new(&[2]).with_values(&[1, 128]).unwrap())
        .build(&mut scope)
        .unwrap();

    let lf_slice = ops::Slice::new()
        .build(lf_input, begin_tensor, size_tensor, &mut scope)
        .unwrap();

    let diff = ops::Sub::new()
        .build(centroids_input, lf_slice, &mut scope)
        .unwrap();

    let squared_diff = ops::Square::new()
        .build(diff, &mut scope)
        .unwrap();

    let axis_tensor = ops::Const::new()
        .dtype(DataType::Int32)
        .value(Tensor::new(&[1]).with_values(&[1]).unwrap())
        .build(&mut scope)
        .unwrap();

    let mean_squared_diff = ops::Mean::new()
        .build(squared_diff, axis_tensor, &mut scope)
        .unwrap();

    let distance = ops::Sqrt::new()
        .build(mean_squared_diff, &mut scope)
        .unwrap();

    let negated_distance = ops::Neg::new()
        .build(distance, &mut scope)
        .unwrap();

    let k_tensor = ops::Const::new()
        .dtype(DataType::Int64)
        .value(centroids.dims()[0] as i64)
        .build(&mut scope)
        .unwrap();

    let top_k = ops::TopKV2::new()
        .build(negated_distance, k_tensor, &mut scope)
        .unwrap();

    let graph = scope.graph();
    let session = Session::new(&SessionOptions::new(), &graph).unwrap();

    let top_k_token = run_args.request_fetch(&top_k, 1);
    session.run(&mut run_args).unwrap();

    let ranked_cluster_labels = run_args.fetch(top_k_token).unwrap();

    ranked_cluster_labels.iter().as_slice().to_vec()
}
