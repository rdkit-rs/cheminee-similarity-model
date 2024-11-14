use flate2::bufread::GzDecoder;
use std::fs::File;
use std::io::BufReader;
use tar::Archive;

fn main() {
    let asset_bytes = reqwest::blocking::get("https://cheminee-models.s3.eu-central-1.amazonaws.com/similarity/similarity-0.1.0.tar.gz")
        .expect("Failed get request")
        .bytes()
        .expect("Failed to retrieve bytes");

    let tar_gz_path = "target/similarity-0.1.0.tar.gz";

    std::fs::write(tar_gz_path, asset_bytes).expect("Failed to write tar file");

    let tar_gz_file = File::open(tar_gz_path).expect("Failed to open tar file");
    let decoder = GzDecoder::new(BufReader::new(tar_gz_file));

    let mut archive = Archive::new(decoder);
    archive.unpack("target").expect("Failed to unpack tar ball");
}
