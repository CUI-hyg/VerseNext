//! training.rs：训练工具链内核（与 `verse_torch/training.py` 同名）。
//!
//! 训练循环高频热点下沉到 Rust：
//! - `grad_norm` / `scale_grads`：梯度全局范数归约 + 裁剪缩放（clip_grad_norm）
//! - `log_softmax_forward` / `log_softmax_backward`：沿最后一维的数值稳定
//!   log_softmax（cross_entropy 的前反向热点），单遍遍历替代 5~6 次
//!   numpy 临时数组，缓存友好。
//!
//! 仅支持 float32；Python 侧在形状/连续/dtype 不符时自动降级原实现。

use numpy::{PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::utils::{flat_to_array, to_slice};

// ---------------------------------------------------------------------------
// 梯度裁剪（clip_grad_norm 内核）
// ---------------------------------------------------------------------------

/// 多个梯度数组的全局 L2 范数（f64 累加，与 NumPy float32 累加同量级）。
#[pyfunction]
pub fn grad_norm(grads: Vec<PyReadonlyArrayDyn<f32>>) -> f32 {
    let mut acc = 0.0f64;
    for g in &grads {
        let v = to_slice(g);
        for &x in v.iter() {
            acc += (x as f64) * (x as f64);
        }
    }
    acc.sqrt() as f32
}

/// 按统一 scale 缩放每个梯度数组，返回新数组（语义同 ``p.grad = p.grad * scale``）。
#[pyfunction]
pub fn scale_grads(
    py: Python<'_>,
    grads: Vec<PyReadonlyArrayDyn<f32>>,
    scale: f32,
) -> PyResult<Vec<Py<PyArrayDyn<f32>>>> {
    let mut out = Vec::with_capacity(grads.len());
    for g in &grads {
        let shape = g.shape().to_vec();
        let v = to_slice(g);
        let scaled: Vec<f32> = v.iter().map(|&x| x * scale).collect();
        out.push(flat_to_array(py, scaled, &shape));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// log_softmax（沿最后一维）
// ---------------------------------------------------------------------------

fn normalize_shape(shape: &[usize]) -> PyResult<(usize, usize)> {
    if shape.is_empty() {
        return Err(PyValueError::new_err(
            "log_softmax: empty shape not supported",
        ));
    }
    let v = *shape.last().expect("non-empty shape");
    let n = shape[..shape.len() - 1].iter().product::<usize>();
    Ok((n, v))
}

/// 数值稳定 log_softmax 前向（沿最后一维）：out = (x - max) - ln(Σ exp(x - max))。
#[pyfunction]
pub fn log_softmax_forward(
    py: Python<'_>,
    x: PyReadonlyArrayDyn<f32>,
) -> PyResult<Py<PyArrayDyn<f32>>> {
    let shape = x.shape().to_vec();
    let (n, v) = normalize_shape(&shape)?;
    let xf = to_slice(&x);
    let mut out = Vec::with_capacity(n * v);
    for r in 0..n {
        let row = &xf[r * v..(r + 1) * v];
        let mut m = f32::NEG_INFINITY;
        for &xr in row.iter() {
            if xr > m {
                m = xr;
            }
        }
        // f64 累加减少 sum 误差；指数用 f64（结果与 NumPy float32 差异 ~1e-7 级）
        let mut se = 0.0f64;
        for &xr in row.iter() {
            se += ((xr - m) as f64).exp();
        }
        let ls = se.ln();
        for &xr in row.iter() {
            out.push(((xr - m) as f64 - ls) as f32);
        }
    }
    Ok(flat_to_array(py, out, &shape))
}

/// log_softmax 反向：dx = grad - exp(out) * Σ grad（沿最后一维）。
#[pyfunction]
pub fn log_softmax_backward(
    py: Python<'_>,
    grad: PyReadonlyArrayDyn<f32>,
    out: PyReadonlyArrayDyn<f32>,
) -> PyResult<Py<PyArrayDyn<f32>>> {
    let shape = grad.shape().to_vec();
    let (n, v) = normalize_shape(&shape)?;
    let gf = to_slice(&grad);
    let of = to_slice(&out);
    if gf.len() != of.len() {
        return Err(PyValueError::new_err(
            "log_softmax_backward: grad/out length mismatch",
        ));
    }
    let mut dx = Vec::with_capacity(n * v);
    for r in 0..n {
        let row_g = &gf[r * v..(r + 1) * v];
        let row_o = &of[r * v..(r + 1) * v];
        let mut sum_g = 0.0f64;
        for &x in row_g.iter() {
            sum_g += x as f64;
        }
        for i in 0..v {
            let s = (row_o[i] as f64).exp();
            dx.push((row_g[i] as f64 - s * sum_g) as f32);
        }
    }
    Ok(flat_to_array(py, dx, &shape))
}
