//! verse_rs：VerseTorch Part1 Rust 内核（pyo3 扩展模块入口）。
//!
//! 实现源码与 Python 同名文件同目录存放（verse_torch/parallel.rs 等），
//! 通过 `#[path]` 引用，保证「同名 .rs 与源码同目录」的工程约定。
//!
//! 构建：`cargo build --release` 后把 `target/release/libverse_rs.so`
//! 复制为 `verse_torch/verse_rs.so`（与源码同目录，保持同名）。

use pyo3::prelude::*;

/// 公共小工具：numpy 视图归一化 + 同 shape 输出构造。
pub(crate) mod utils {
    use std::borrow::Cow;

    use numpy::ndarray::{ArrayD, IxDyn};
    use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods};
    use pyo3::prelude::*;

    /// 取 ndarray 视图：连续则零拷贝借用，否则拷贝为连续 Vec。
    pub fn to_slice<'a>(a: &'a PyReadonlyArrayDyn<f32>) -> Cow<'a, [f32]> {
        if a.is_c_contiguous() {
            Cow::Borrowed(a.as_slice().expect("contiguous slice"))
        } else {
            Cow::Owned(a.as_array().iter().copied().collect())
        }
    }

    /// 用展平数据 + shape 构造同形状输出数组。
    pub fn flat_to_array(
        py: Python<'_>,
        flat: Vec<f32>,
        shape: &[usize],
    ) -> Py<PyArrayDyn<f32>> {
        let arr = ArrayD::from_shape_vec(IxDyn(shape), flat)
            .expect("flattened length must match shape");
        arr.into_pyarray(py).unbind()
    }
}

#[path = "../verse_torch/parallel.rs"]
pub mod parallel;

#[path = "../verse_torch/optim.rs"]
pub mod optim;

#[path = "../verse_torch/training.rs"]
pub mod training;

/// Python 侧入口：`from .verse_rs import batched_matmul, ...`。
#[pymodule]
fn verse_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parallel::batched_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(parallel::default_threads, m)?)?;
    m.add_function(wrap_pyfunction!(optim::adam_step, m)?)?;
    m.add_function(wrap_pyfunction!(optim::nadamw_step, m)?)?;
    m.add_function(wrap_pyfunction!(optim::sgd_step, m)?)?;
    m.add_function(wrap_pyfunction!(optim::rmsprop_step, m)?)?;
    m.add_function(wrap_pyfunction!(optim::lion_step, m)?)?;
    m.add_function(wrap_pyfunction!(training::grad_norm, m)?)?;
    m.add_function(wrap_pyfunction!(training::scale_grads, m)?)?;
    m.add_function(wrap_pyfunction!(training::log_softmax_forward, m)?)?;
    m.add_function(wrap_pyfunction!(training::log_softmax_backward, m)?)?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}
