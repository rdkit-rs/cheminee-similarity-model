use csv::ReaderBuilder;
use ndarray::Array2;
use std::fs::File;
use bitvec::vec::BitVec;
use tensorflow::{DataType, Graph, ops, SavedModelBundle, Scope, Session, SessionOptions, SessionRunArgs, Tensor};

pub fn load_encoder_model() -> (SavedModelBundle, Graph) {
    let model_dir = "assets/vae_encoder";
    let session_options = SessionOptions::new();
    let mut graph = Graph::new();
    let saved_model = SavedModelBundle::load(&session_options, vec!["serve"], &mut graph, model_dir).unwrap();
    (saved_model, graph)
}

pub fn encode(model: &SavedModelBundle, graph: &Graph, input_data: &BitVec) -> Tensor<f32> {
    let input_data = input_data.iter().map(|b| if *b {1} else {0}).collect::<Vec<i64>>();
    let input_tensor = Tensor::new(&[1, input_data.len() as u64]).with_values(&input_data).unwrap();

    let input_operation = graph.operation_by_name("serving_default_dense_input").unwrap().unwrap();
    let output_operation = graph.operation_by_name("StatefulPartitionedCall").unwrap().unwrap();

    let mut run_args = SessionRunArgs::new();
    run_args.add_feed(&input_operation, 0, &input_tensor);

    let output_token = run_args.request_fetch(&output_operation, 0);
    model.session.run(&mut run_args).unwrap();

    let output_tensor = run_args.fetch(output_token).unwrap();
    let output_slice = &output_tensor.iter().as_slice()[..128];
    let lf_tensor = Tensor::new(&[1, 128]).with_values(output_slice).unwrap();

    lf_tensor
}

pub fn load_cluster_centroids() -> Tensor<f32> {
    let file = File::open("assets/lf_kmeans_10k_centroids_20241025.csv").unwrap();
    let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

    let mut data: Vec<Vec<f32>> = Vec::new();
    for result in rdr.records() {
        let record = result.unwrap();
        let row: Vec<f32> = record.iter()
            .map(|s| s.parse().unwrap())
            .collect();
        data.push(row);
    }

    let array: Array2<f32> = Array2::from_shape_vec((data.len(), data[0].len()), data.concat()).unwrap();

    let tensor = Tensor::new(&[array.shape()[0] as u64, array.shape()[1] as u64])
        .with_values(array.as_slice().unwrap()).unwrap();

    tensor
}

pub fn assign_cluster_labels(centroids: &Tensor<f32>, lf_array: &Tensor<f32>) -> Tensor<i32> {
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

    let diff = ops::Sub::new()
        .build(centroids_input, lf_input, &mut scope)
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

    ranked_cluster_labels
}
