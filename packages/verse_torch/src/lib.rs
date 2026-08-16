//! verse_rs：VerseTorch Part1 Rust 内核（pyo3 扩展模块入口）。
//!
//! 实现源码与 Python 同名文件同目录存放（verse_torch/parallel.rs 等），
//! 通过 `#[path]` 引用，保证「同名 .rs 与源码同目录」的工程约定。
//!
//! 构建：`cargo build --release` 后把 `target/release/libverse_rs.so`
//! 复制为 `verse_torch/verse_rs.so`（与源码同目录，保持同名）。

use pyo3::prelude::*;

#[path = "../verse_torch/parallel.rs"]
pub mod parallel;

/// Python 侧入口：`from .verse_rs import batched_matmul, ...`。
#[pymodule]
fn verse_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parallel::batched_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(parallel::default_threads, m)?)?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}
