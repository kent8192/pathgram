use super::{read_jsonl, LoadError, Loader};
use crate::eval::instance::SweInstance;
use std::path::Path;

pub struct SweGymLoader;

impl Loader for SweGymLoader {
    fn load(&self, path: &Path) -> Result<Vec<SweInstance>, LoadError> {
        read_jsonl(path)
    }

    fn dataset_name(&self) -> &'static str {
        "SWE-Gym"
    }
}
