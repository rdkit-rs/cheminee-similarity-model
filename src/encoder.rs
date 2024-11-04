use bitvec::vec::BitVec;
use tensorflow::{Graph, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};

pub fn load_encoder_model() -> (SavedModelBundle, Graph) {
    let model_dir = "assets/vae_encoder";
    let session_options = SessionOptions::new();
    let mut graph = Graph::new();
    let saved_model = SavedModelBundle::load(&session_options, vec!["serve"], &mut graph, model_dir).unwrap();
    (saved_model, graph)
}

pub fn encode(model: SavedModelBundle, graph: Graph, input_data: BitVec) -> Tensor<f32> {
    let input_data = input_data.iter().map(|b| if *b {1} else {0}).collect::<Vec<i64>>();
    let input_tensor = Tensor::new(&[1, input_data.len() as u64]).with_values(&input_data).unwrap();

    let input_operation = graph.operation_by_name("serving_default_dense_input").unwrap().unwrap();
    let output_operation = graph.operation_by_name("StatefulPartitionedCall").unwrap().unwrap();

    let mut run_args = SessionRunArgs::new();
    run_args.add_feed(&input_operation, 0, &input_tensor);

    let output_token = run_args.request_fetch(&output_operation, 0);
    model.session.run(&mut run_args).unwrap();

    let output_tensor = run_args.fetch(output_token).unwrap();

    output_tensor
}