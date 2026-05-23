#![cfg(feature = "eval")]

use pathgram::eval::loaders::{
    swe_bench_lite::SweBenchLiteLoader, swe_gym::SweGymLoader, Loader,
};
use std::path::PathBuf;

fn fixture_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/eval/fixtures/swe_bench_lite_subset.jsonl");
    p
}

#[test]
fn swe_bench_lite_loader_reads_jsonl_fixture() {
    let loader = SweBenchLiteLoader;
    let instances = loader.load(&fixture_path()).expect("load fixture");
    assert_eq!(instances.len(), 5);
    assert_eq!(instances[0].instance_id, "django__django-11099");
    assert_eq!(instances[3].instance_id, "flask-test-1");
    assert!(instances[3].is_synthetic_fixture());
    assert_eq!(loader.dataset_name(), "SWE-bench_Lite");
}

#[test]
fn swe_gym_loader_reuses_lite_schema() {
    let loader = SweGymLoader;
    let instances = loader.load(&fixture_path()).expect("load fixture");
    assert_eq!(instances.len(), 5);
    assert_eq!(loader.dataset_name(), "SWE-Gym");
}
