//! optim.rs：优化器 step 更新内核（与 `verse_torch/optim.py` 同名）。
//!
//! 将各优化器逐参数、逐元素的更新循环下沉到 Rust：单次遍历完成
//! m/v/buf 状态更新 + 参数更新，避免 Python 侧每参数 5~8 次
//! numpy 临时数组分配。仅支持 float32（连续/非连续均可，内部归一）。
//!
//! 数学语义与 Python 实现严格对齐（IEEE754 逐元素，顺序一致）：
//! - `adam_step`：Adam（coupled wd）/ AdamW（decoupled wd）共用
//! - `nadamw_step`：NAdamW（Nesterov 前瞻一阶矩）
//! - `sgd_step`：SGD（momentum / dampening / nesterov / wd）
//! - `rmsprop_step`：RMSProp（momentum / centered / wd）
//! - `lion_step`：Lion（sign 更新 + 解耦 wd）

use std::borrow::Cow;

use numpy::{PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::utils::{flat_to_array, to_slice};
/// 校验一组同 shape 数组的展平长度。
fn check_all_len(
    plen: usize,
    lens: &[(&str, usize)],
) -> PyResult<()> {
    for (name, len) in lens {
        if *len != plen {
            return Err(PyValueError::new_err(format!(
                "shape mismatch: param.len={} {name}.len={}",
                plen, len
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Adam / AdamW
// ---------------------------------------------------------------------------

/// 逐元素 Adam 内核（coupled/decoupled weight decay 统一）。
///
/// decoupled=false（Adam）:  g_i = g + wd * p
/// decoupled=true （AdamW）: p_new = (p - lr*m_hat/(sqrt(v_hat)+eps)) * (1 - lr*wd)
fn adam_kernel(
    p: &[f32], g: &[f32], m: &mut [f32], v: &mut [f32],
    lr: f32, beta1: f32, beta2: f32, eps: f32,
    bc1: f32, bc2: f32, wd: f32, decoupled: bool,
) -> Vec<f32> {
    let mut pn = Vec::with_capacity(p.len());
    for i in 0..p.len() {
        let gi = if decoupled { g[i] } else { g[i] + wd * p[i] };
        let mi = beta1 * m[i] + (1.0 - beta1) * gi;
        let vi = beta2 * v[i] + (1.0 - beta2) * gi * gi;
        m[i] = mi;
        v[i] = vi;
        let m_hat = mi / bc1;
        let v_hat = vi / bc2;
        let update = lr * m_hat / (v_hat.sqrt() + eps);
        pn.push(if decoupled {
            (p[i] - update) * (1.0 - lr * wd)
        } else {
            p[i] - update
        });
    }
    pn
}

/// Adam / AdamW 一步更新。返回 (param', m', v')。
#[pyfunction]
#[pyo3(signature = (param, grad, m, v, lr, beta1, beta2, eps, bc1, bc2, weight_decay, decoupled))]
#[allow(clippy::too_many_arguments)]
pub fn adam_step(
    py: Python<'_>,
    param: PyReadonlyArrayDyn<f32>,
    grad: PyReadonlyArrayDyn<f32>,
    m: PyReadonlyArrayDyn<f32>,
    v: PyReadonlyArrayDyn<f32>,
    lr: f32, beta1: f32, beta2: f32, eps: f32,
    bc1: f32, bc2: f32, weight_decay: f32, decoupled: bool,
) -> PyResult<(
    Py<PyArrayDyn<f32>>,
    Py<PyArrayDyn<f32>>,
    Py<PyArrayDyn<f32>>,
)> {
    let shape = param.shape().to_vec();
    let plen = shape.iter().product::<usize>();
    let (p, g) = (to_slice(&param), to_slice(&grad));
    let (m, v) = (to_slice(&m).into_owned(), to_slice(&v).into_owned());
    check_all_len(
        plen,
        &[
            ("param", p.len()),
            ("grad", g.len()),
            ("m", m.len()),
            ("v", v.len()),
        ],
    )?;
    let mut mw = m;
    let mut vw = v;
    let pn = adam_kernel(&p, &g, &mut mw, &mut vw, lr, beta1, beta2, eps, bc1, bc2, weight_decay, decoupled);
    Ok((
        flat_to_array(py, pn, &shape),
        flat_to_array(py, mw, &shape),
        flat_to_array(py, vw, &shape),
    ))
}

// ---------------------------------------------------------------------------
// NAdamW
// ---------------------------------------------------------------------------

/// NAdamW：m_nesterov = beta1*m' + (1-beta1)*g，解耦 wd。
fn nadamw_kernel(
    p: &[f32], g: &[f32], m: &mut [f32], v: &mut [f32],
    lr: f32, beta1: f32, beta2: f32, eps: f32,
    bc1: f32, bc2: f32, wd: f32,
) -> Vec<f32> {
    let mut pn = Vec::with_capacity(p.len());
    for i in 0..p.len() {
        let gi = g[i];
        let mi = beta1 * m[i] + (1.0 - beta1) * gi;
        let vi = beta2 * v[i] + (1.0 - beta2) * gi * gi;
        m[i] = mi;
        v[i] = vi;
        let m_nesterov = beta1 * mi + (1.0 - beta1) * gi;
        let m_hat = m_nesterov / bc1;
        let v_hat = vi / bc2;
        let update = lr * m_hat / (v_hat.sqrt() + eps);
        pn.push((p[i] - update) * (1.0 - lr * wd));
    }
    pn
}

/// NAdamW 一步更新。返回 (param', m', v')。
#[pyfunction]
#[pyo3(signature = (param, grad, m, v, lr, beta1, beta2, eps, bc1, bc2, weight_decay))]
#[allow(clippy::too_many_arguments)]
pub fn nadamw_step(
    py: Python<'_>,
    param: PyReadonlyArrayDyn<f32>,
    grad: PyReadonlyArrayDyn<f32>,
    m: PyReadonlyArrayDyn<f32>,
    v: PyReadonlyArrayDyn<f32>,
    lr: f32, beta1: f32, beta2: f32, eps: f32,
    bc1: f32, bc2: f32, weight_decay: f32,
) -> PyResult<(
    Py<PyArrayDyn<f32>>,
    Py<PyArrayDyn<f32>>,
    Py<PyArrayDyn<f32>>,
)> {
    let shape = param.shape().to_vec();
    let plen = shape.iter().product::<usize>();
    let (p, g) = (to_slice(&param), to_slice(&grad));
    let (m, v) = (to_slice(&m).into_owned(), to_slice(&v).into_owned());
    check_all_len(
        plen,
        &[
            ("param", p.len()),
            ("grad", g.len()),
            ("m", m.len()),
            ("v", v.len()),
        ],
    )?;
    let mut mw = m;
    let mut vw = v;
    let pn = nadamw_kernel(&p, &g, &mut mw, &mut vw, lr, beta1, beta2, eps, bc1, bc2, weight_decay);
    Ok((
        flat_to_array(py, pn, &shape),
        flat_to_array(py, mw, &shape),
        flat_to_array(py, vw, &shape),
    ))
}

// ---------------------------------------------------------------------------
// SGD
// ---------------------------------------------------------------------------

/// SGD 逐元素内核。
///
/// buf 为 Option：None 表示首次（Python 语义为 buf = g_wd.copy()）。
fn sgd_kernel(
    p: &[f32], g: &[f32], buf: Option<&[f32]>,
    lr: f32, momentum: f32, dampening: f32, wd: f32, nesterov: bool,
) -> (Vec<f32>, Option<Vec<f32>>) {
    let mut pn = Vec::with_capacity(p.len());
    let mut buf_new: Option<Vec<f32>> = if momentum != 0.0 {
        Some(Vec::with_capacity(p.len()))
    } else {
        None
    };
    for i in 0..p.len() {
        let g_wd = g[i] + wd * p[i];
        if let Some(bn) = buf_new.as_mut() {
            let b = match buf {
                Some(b) => momentum * b[i] + (1.0 - dampening) * g_wd,
                None => g_wd, // 首次：buf = g_wd.copy()
            };
            bn.push(b);
            let update = if nesterov {
                g_wd + momentum * b
            } else {
                b
            };
            pn.push(p[i] - lr * update);
        } else {
            pn.push(p[i] - lr * g_wd);
        }
    }
    (pn, buf_new)
}

/// SGD 一步更新。返回 (param', buf')；无 momentum 时 buf' 为 None。
#[pyfunction]
#[pyo3(signature = (param, grad, buf, lr, momentum, dampening, weight_decay, nesterov))]
pub fn sgd_step(
    py: Python<'_>,
    param: PyReadonlyArrayDyn<f32>,
    grad: PyReadonlyArrayDyn<f32>,
    buf: Option<PyReadonlyArrayDyn<f32>>,
    lr: f32, momentum: f32, dampening: f32, weight_decay: f32, nesterov: bool,
) -> PyResult<(
    Py<PyArrayDyn<f32>>,
    Option<Py<PyArrayDyn<f32>>>,
)> {
    let shape = param.shape().to_vec();
    let plen = shape.iter().product::<usize>();
    let (p, g) = (to_slice(&param), to_slice(&grad));
    let b: Option<Cow<'_, [f32]>> = buf.as_ref().map(to_slice);
    let mut lens = vec![("param", p.len()), ("grad", g.len())];
    if let Some(Cow::Borrowed(b)) = &b {
        lens.push(("buf", b.len()));
    }
    check_all_len(plen, &lens)?;
    let b_ref: Option<&[f32]> = b.as_deref();
    let (pn, buf_new) = sgd_kernel(&p, &g, b_ref, lr, momentum, dampening, weight_decay, nesterov);
    let p_arr = flat_to_array(py, pn, &shape);
    let b_arr = buf_new.map(|bn| flat_to_array(py, bn, &shape));
    Ok((p_arr, b_arr))
}

// ---------------------------------------------------------------------------
// RMSProp
// ---------------------------------------------------------------------------

/// RMSProp 逐元素内核（支持 momentum / centered）。
fn rmsprop_kernel(
    p: &[f32], g: &[f32], v: &mut [f32],
    buf: Option<&[f32]>, avg: Option<&[f32]>,
    lr: f32, alpha: f32, eps: f32, wd: f32,
    momentum: f32, centered: bool,
) -> (Vec<f32>, Vec<f32>, Option<Vec<f32>>, Option<Vec<f32>>) {
    let mut pn = Vec::with_capacity(p.len());
    let mut vn = Vec::with_capacity(p.len());
    let mut buf_new: Option<Vec<f32>> = if momentum != 0.0 {
        Some(Vec::with_capacity(p.len()))
    } else {
        None
    };
    let mut avg_new: Option<Vec<f32>> = if centered {
        Some(Vec::with_capacity(p.len()))
    } else {
        None
    };
    // 循环外取借用：仅启用时解引用
    let buf_slice: Option<&[f32]> = if momentum != 0.0 {
        Some(buf.expect("momentum requires buf"))
    } else {
        None
    };
    let avg_slice: Option<&[f32]> = if centered {
        Some(avg.expect("centered requires avg"))
    } else {
        None
    };
    for i in 0..p.len() {
        let g_wd = g[i] + wd * p[i];
        let vi = alpha * v[i] + (1.0 - alpha) * g_wd * g_wd;
        vn.push(vi);
        let denom = match avg_slice {
            Some(a) => {
                let ai = alpha * a[i] + (1.0 - alpha) * g_wd;
                if let Some(an) = avg_new.as_mut() {
                    an.push(ai);
                }
                (vi - ai * ai).sqrt() + eps
            }
            None => vi.sqrt() + eps,
        };
        match buf_slice {
            Some(b) => {
                let bi = momentum * b[i] + g_wd;
                if let Some(bn) = buf_new.as_mut() {
                    bn.push(bi);
                }
                pn.push(p[i] - lr * bi / denom);
            }
            None => pn.push(p[i] - lr * g_wd / denom),
        }
    }
    (pn, vn, buf_new, avg_new)
}

/// RMSProp 一步更新。返回 (param', v', buf'?, avg'?)。
#[pyfunction]
#[pyo3(signature = (param, grad, v, buf, avg, lr, alpha, eps, weight_decay, momentum, centered))]
#[allow(clippy::too_many_arguments)]
pub fn rmsprop_step(
    py: Python<'_>,
    param: PyReadonlyArrayDyn<f32>,
    grad: PyReadonlyArrayDyn<f32>,
    v: PyReadonlyArrayDyn<f32>,
    buf: Option<PyReadonlyArrayDyn<f32>>,
    avg: Option<PyReadonlyArrayDyn<f32>>,
    lr: f32, alpha: f32, eps: f32, weight_decay: f32,
    momentum: f32, centered: bool,
) -> PyResult<(
    Py<PyArrayDyn<f32>>,
    Py<PyArrayDyn<f32>>,
    Option<Py<PyArrayDyn<f32>>>,
    Option<Py<PyArrayDyn<f32>>>,
)> {
    let shape = param.shape().to_vec();
    let plen = shape.iter().product::<usize>();
    let (p, g) = (to_slice(&param), to_slice(&grad));
    let v = to_slice(&v).into_owned();
    let b: Option<Cow<'_, [f32]>> = buf.as_ref().map(to_slice);
    let a: Option<Cow<'_, [f32]>> = avg.as_ref().map(to_slice);
    // 防御性校验：缺失必选状态返回错误（避免 panic 逃逸为 PanicException）
    if momentum != 0.0 && b.is_none() {
        return Err(PyValueError::new_err("rmsprop_step: momentum requires buf"));
    }
    if centered && a.is_none() {
        return Err(PyValueError::new_err("rmsprop_step: centered requires avg"));
    }
    let mut lens = vec![("param", p.len()), ("grad", g.len()), ("v", v.len())];
    if let Some(Cow::Borrowed(b)) = &b {
        lens.push(("buf", b.len()));
    }
    if let Some(Cow::Borrowed(a)) = &a {
        lens.push(("avg", a.len()));
    }
    check_all_len(plen, &lens)?;
    let mut vw = v;
    let (pn, vn, buf_new, avg_new) = rmsprop_kernel(
        &p, &g, &mut vw, b.as_deref(), a.as_deref(),
        lr, alpha, eps, weight_decay, momentum, centered,
    );
    Ok((
        flat_to_array(py, pn, &shape),
        flat_to_array(py, vn, &shape),
        buf_new.map(|bn| flat_to_array(py, bn, &shape)),
        avg_new.map(|an| flat_to_array(py, an, &shape)),
    ))
}

// ---------------------------------------------------------------------------
// Lion
// ---------------------------------------------------------------------------

/// 与 ``np.sign`` 语义一致的符号函数：±0.0 -> 0.0，NaN -> NaN。
/// （Rust 内置 ``f32::signum`` 对 +0.0 返回 +1.0，与 numpy 不一致。）
fn np_sign(x: f32) -> f32 {
    if x.is_nan() {
        f32::NAN
    } else if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Lion 逐元素内核。
fn lion_kernel(
    p: &[f32], g: &[f32], m: &mut [f32],
    lr: f32, beta1: f32, beta2: f32, wd: f32,
) -> Vec<f32> {
    let mut pn = Vec::with_capacity(p.len());
    for i in 0..p.len() {
        let gi = g[i];
        let mi = m[i];
        let update = mi * beta1 + gi * (1.0 - beta1);
        let s = np_sign(update);
        let pi = p[i] - lr * s;
        pn.push(if wd != 0.0 {
            pi - lr * wd * pi
        } else {
            pi
        });
        m[i] = beta2 * mi + (1.0 - beta2) * gi;
    }
    pn
}

/// Lion 一步更新。返回 (param', m')。
#[pyfunction]
#[pyo3(signature = (param, grad, m, lr, beta1, beta2, weight_decay))]
pub fn lion_step(
    py: Python<'_>,
    param: PyReadonlyArrayDyn<f32>,
    grad: PyReadonlyArrayDyn<f32>,
    m: PyReadonlyArrayDyn<f32>,
    lr: f32, beta1: f32, beta2: f32, weight_decay: f32,
) -> PyResult<(
    Py<PyArrayDyn<f32>>,
    Py<PyArrayDyn<f32>>,
)> {
    let shape = param.shape().to_vec();
    let plen = shape.iter().product::<usize>();
    let (p, g) = (to_slice(&param), to_slice(&grad));
    let m = to_slice(&m).into_owned();
    check_all_len(plen, &[("param", p.len()), ("grad", g.len()), ("m", m.len())])?;
    let mut mw = m;
    let pn = lion_kernel(&p, &g, &mut mw, lr, beta1, beta2, weight_decay);
    Ok((flat_to_array(py, pn, &shape), flat_to_array(py, mw, &shape)))
}
