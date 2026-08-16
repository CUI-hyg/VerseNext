//! verse_torch.parallel 的 Rust 数值内核（与 parallel.py 同名同目录）。
//!
//! 覆盖 parallel.py 的 CPU 热点：批量矩阵乘法（batched matmul）。
//! - `batched_matmul`：支持 (B,M,K)x(K,N)、(B,M,K)x(B,K,N)、(M,K)x(K,N)，
//!   等价于 numpy `np.matmul`（数值容差 1e-4 内一致）。
//! - 并行模型：rayon 线程池按 batch 分片（替代 Python multiprocessing 进程池，
//!   省去 fork/pickle/进程创建开销；B 共享零拷贝）。
//! - `default_threads`：与 parallel.py `_default_n_workers()` 一致
//!   （CPU 核数一半，至少 1）。
//!
//! Python 侧 parallel.py 保留完整 API 与 multiprocessing 降级路径：
//! .so 可用时优先走本内核，不可用时回退原实现。

use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// 工具
// ---------------------------------------------------------------------------

/// 默认线程数：CPU 核数一半（至少 1），与 parallel.py 语义一致。
#[pyfunction]
pub fn default_threads() -> usize {
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2);
    std::cmp::max(1, cpus / 2)
}

/// 把 numpy 输入复制为连续 f32 Vec（处理非连续内存）。
fn to_f32_vec(arr: &PyReadonlyArrayDyn<f32>) -> Vec<f32> {
    let v: Vec<f32> = if arr.is_c_contiguous() {
        arr.as_slice().expect("contiguous slice").to_vec()
    } else {
        arr.as_array().iter().copied().collect()
    };
    v
}

/// 连续输入零拷贝借用为 `&[f32]`；非连续时回退拷贝（保证计算逻辑一致）。
enum Borrowed<'py> {
    Slice(&'py [f32]),
    Owned(Vec<f32>),
}

impl<'py> Borrowed<'py> {
    fn as_ref(&self) -> &[f32] {
        match self {
            Borrowed::Slice(s) => s,
            Borrowed::Owned(v) => v,
        }
    }
}

fn borrow_f32<'py>(arr: &'py PyReadonlyArrayDyn<f32>) -> Borrowed<'py> {
    if arr.is_c_contiguous() {
        Borrowed::Slice(arr.as_slice().expect("contiguous slice"))
    } else {
        Borrowed::Owned(arr.as_array().iter().copied().collect())
    }
}

/// 分片：把 batch 切成 ≈n_threads 段（测试辅助）。
#[cfg(test)]
fn split_batch(batch: usize, n_threads: usize) -> Vec<(usize, usize)> {
    let k = if n_threads > batch { batch } else { n_threads };
    if k < 1 {
        return vec![(0, batch)];
    }
    let chunk = (batch + k - 1) / k;
    let mut out = Vec::with_capacity(k);
    let mut i = 0;
    while i < batch {
        let end = std::cmp::min(i + chunk, batch);
        out.push((i, end));
        i = end;
    }
    out
}

// ---------------------------------------------------------------------------
// GEMM 内核（f32，行主序，基于 matrixmultiply 的 SIMD 优化实现）
// ---------------------------------------------------------------------------

/// 单 GEMM：C(M,N) = A(M,K) x B(K,N)，行主序，单线程 SIMD 优化。
fn gemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            n as isize,
            1,
            0.0,
            c.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

/// 带偏置的 GEMM：C = A x B + bias（bias 广播到每一行；可选）。
fn gemm_bias(
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    gemm(a, b, c, m, k, n);
    if let Some(bi) = bias {
        for i in 0..m {
            let c_row = &mut c[i * n..(i + 1) * n];
            for (cj, &bv) in c_row.iter_mut().zip(bi.iter()) {
                *cj += bv;
            }
        }
    }
}

/// 分块行并行 GEMM（2D x 2D，batch=1 时用）。
///
/// 把 M 切成 n_threads 段，每段一个独立 sgemm（SIMD 微内核），rayon 并行；
/// C 的行段互不重叠，无写冲突。
fn gemm_parallel(
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    n_threads: usize,
) {
    if m < 8 || n_threads <= 1 {
        gemm_bias(a, b, bias, out, m, k, n);
        return;
    }
    // 行段递归分治：每层 split_at_mut 安全切分 C，rayon::join 并行两半；
    // 深度到 ceil(log2(n_threads)) 后串行执行段内 sgemm。
    let target_depth = n_threads.next_power_of_two().trailing_zeros() as usize;
    fn rec(
        a: &[f32],
        b: &[f32],
        bias: Option<&[f32]>,
        out: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
        depth: usize,
        max_depth: usize,
    ) {
        if m <= 16 || depth >= max_depth {
            gemm_bias(a, b, bias, out, m, k, n);
            return;
        }
        let mid = m / 2;
        let (c_lo, c_hi) = out.split_at_mut(mid * n);
        rayon::join(
            || rec(a, b, bias, c_lo, mid, k, n, depth + 1, max_depth),
            || {
                rec(
                    &a[mid * k..],
                    b,
                    bias,
                    c_hi,
                    m - mid,
                    k,
                    n,
                    depth + 1,
                    max_depth,
                )
            },
        );
    }
    rec(a, b, bias, out, m, k, n, 0, target_depth);
}

// ---------------------------------------------------------------------------
// batched matmul
// ---------------------------------------------------------------------------

enum Shape {
    // (M,K) x (K,N) -> (M,N)
    Mmul {
        m: usize,
        k: usize,
        n: usize,
    },
    // (B,M,K) x (K,N) -> (B,M,N)
    Batched {
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    },
    // (B,M,K) x (B,K,N) -> (B,M,N)
    Paired {
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    },
}

/// 校验 GEMM 维度是否匹配，返回 err 说明。
fn check_dims(a_shape: &[usize], b_shape: &[usize]) -> Result<Shape, String> {
    match (a_shape.len(), b_shape.len()) {
        (2, 2) => {
            if a_shape[1] != b_shape[0] {
                return Err(format!(
                    "matmul 维度不匹配: A(M,K)={}x{} B(K,N)={}x{}",
                    a_shape[0], a_shape[1], b_shape[0], b_shape[1]
                ));
            }
            Ok(Shape::Mmul {
                m: a_shape[0],
                k: a_shape[1],
                n: b_shape[1],
            })
        }
        (3, 2) => {
            if a_shape[2] != b_shape[0] {
                return Err(format!(
                    "matmul 维度不匹配: A(B,M,K)={}x{}x{} B(K,N)={}x{}",
                    a_shape[0], a_shape[1], a_shape[2], b_shape[0], b_shape[1]
                ));
            }
            Ok(Shape::Batched {
                batch: a_shape[0],
                m: a_shape[1],
                k: a_shape[2],
                n: b_shape[1],
            })
        }
        (3, 3) => {
            if a_shape[2] != b_shape[1] || a_shape[0] != b_shape[0] {
                return Err(format!(
                    "matmul 维度不匹配: A(B,M,K)={}x{}x{} B(B,K,N)={}x{}x{}",
                    a_shape[0], a_shape[1], a_shape[2], b_shape[0], b_shape[1], b_shape[2]
                ));
            }
            Ok(Shape::Paired {
                batch: a_shape[0],
                m: a_shape[1],
                k: a_shape[2],
                n: b_shape[2],
            })
        }
        _ => Err(format!(
            "unsupported matmul shapes: A.ndim={} B.ndim={}（支持 2D/3D）",
            a_shape.len(),
            b_shape.len()
        )),
    }
}

/// 批量矩阵乘法（Rust 内核）。
///
/// 参数：
/// - `a`: (B,M,K) 或 (M,K) 的 float32 ndarray
/// - `b`: (K,N) 或 (B,K,N) 的 float32 ndarray
/// - `bias`: 可选 (N,) float32（对输出每行广播；等价于 out + bias）
/// - `n_threads`: 线程数；None 用 `default_threads()`
///
/// 返回：float32 ndarray，shape 与 numpy `np.matmul(a, b)` 一致。
#[pyfunction]
#[pyo3(signature = (a, b, bias=None, n_threads=None))]
pub fn batched_matmul(
    py: Python<'_>,
    a: PyReadonlyArrayDyn<f32>,
    b: PyReadonlyArrayDyn<f32>,
    bias: Option<PyReadonlyArrayDyn<f32>>,
    n_threads: Option<usize>,
) -> PyResult<Py<PyArrayDyn<f32>>> {
    let a_shape: Vec<usize> = a.shape().to_vec();
    let b_shape: Vec<usize> = b.shape().to_vec();

    // 维度校验（先不拷贝数据）
    let shape = check_dims(&a_shape, &b_shape).map_err(PyValueError::new_err)?;

    let threads = match n_threads {
        Some(t) => std::cmp::max(1, t),
        None => default_threads(),
    };

    // bias 校验
    let bias_vec = match &bias {
        Some(bi) => {
            if bi.len() != b_shape.last().copied().unwrap_or(0) {
                return Err(PyValueError::new_err(
                    "bias 长度必须等于输出 N 维",
                ));
            }
            Some(to_f32_vec(bi))
        }
        None => None,
    };

    let out_shape: Vec<usize> = match &shape {
        Shape::Mmul { m: _, k: _, n } => vec![a_shape[0], *n],
        Shape::Batched {
            batch: _,
            m: _,
            k: _,
            n,
        } => vec![a_shape[0], a_shape[1], *n],
        Shape::Paired {
            batch: _,
            m: _,
            k: _,
            n,
        } => vec![a_shape[0], a_shape[1], *n],
    };

    let n_out: usize = out_shape.iter().product();
    let mut out = vec![0.0f32; n_out];

    match shape {
        Shape::Mmul { m, k, n } => {
            let a_vec = to_f32_vec(&a);
            let b_vec = to_f32_vec(&b);
            gemm_parallel(
                &a_vec, &b_vec, bias_vec.as_deref(), &mut out, m, k, n, threads,
            );
        }
        Shape::Batched { batch, m, k, n } => {
            let a_borrow = borrow_f32(&a);
            let b_borrow = borrow_f32(&b);
            let a_vec = a_borrow.as_ref();
            let b_vec = b_borrow.as_ref();
            // 每个 batch：a[bi] (M,K) x b (K,N) -> out[bi] (M,N)
            let a_mk = m * k;
            let c_mn = m * n;
            if batch == 1 || threads == 1 {
                for bi in 0..batch {
                    let a_slice = &a_vec[bi * a_mk..(bi + 1) * a_mk];
                    let c_slice = &mut out[bi * c_mn..(bi + 1) * c_mn];
                    gemm_bias(a_slice, b_vec, bias_vec.as_deref(), c_slice, m, k, n);
                }
            } else {
                // B 共享（只读零拷贝引用），batch 并行
                let bias_ref: Option<&[f32]> = bias_vec.as_deref();
                out.par_chunks_mut(c_mn)
                    .enumerate()
                    .for_each(|(bi, c_slice)| {
                        let a_slice = &a_vec[bi * a_mk..(bi + 1) * a_mk];
                        gemm_bias(a_slice, b_vec, bias_ref, c_slice, m, k, n);
                    });
            }
        }
        Shape::Paired { batch, m, k, n } => {
            let a_borrow = borrow_f32(&a);
            let b_borrow = borrow_f32(&b);
            let a_vec = a_borrow.as_ref();
            let b_vec = b_borrow.as_ref();
            let a_mk = m * k;
            let b_kn = k * n;
            let c_mn = m * n;
            if batch == 1 || threads == 1 {
                for bi in 0..batch {
                    let a_slice = &a_vec[bi * a_mk..(bi + 1) * a_mk];
                    let b_slice = &b_vec[bi * b_kn..(bi + 1) * b_kn];
                    let c_slice = &mut out[bi * c_mn..(bi + 1) * c_mn];
                    gemm_bias(a_slice, b_slice, bias_vec.as_deref(), c_slice, m, k, n);
                }
            } else {
                let bias_ref: Option<&[f32]> = bias_vec.as_deref();
                out.par_chunks_mut(c_mn)
                    .enumerate()
                    .for_each(|(bi, c_slice)| {
                        let a_slice = &a_vec[bi * a_mk..(bi + 1) * a_mk];
                        let b_slice = &b_vec[bi * b_kn..(bi + 1) * b_kn];
                        gemm_bias(a_slice, b_slice, bias_ref, c_slice, m, k, n);
                    });
            }
        }
    }

    // 以正确 shape 构造输出（Vec -> ndarray -> PyArrayDyn）
    let arr = numpy::ndarray::Array::from_shape_vec(out_shape, out)
        .map_err(|e| PyValueError::new_err(format!("shape error: {e}")))?;
    Ok(arr.into_pyarray(py).unbind())
}

// ---------------------------------------------------------------------------
// Rust 单元测试（cargo test）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn ref_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for kk in 0..k {
                let av = a[i * k + kk];
                if av != 0.0 {
                    for j in 0..n {
                        c[i * n + j] += av * b[kk * n + j];
                    }
                }
            }
        }
        c
    }

    #[test]
    fn gemm_matches_reference() {
        let m = 7;
        let k = 5;
        let n = 9;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.5).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.25).collect();
        let mut c = vec![0.0; m * n];
        gemm(&a, &b, &mut c, m, k, n);
        let r = ref_matmul(&a, &b, m, k, n);
        for (x, y) in c.iter().zip(r.iter()) {
            assert!((x - y).abs() < 1e-3, "mismatch {x} vs {y}");
        }
    }

    #[test]
    fn gemm_bias_broadcast() {
        let m = 3;
        let k = 2;
        let n = 4;
        let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32).collect();
        let bias = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut c = vec![0.0; m * n];
        gemm_bias(&a, &b, Some(&bias), &mut c, m, k, n);
        let mut r = ref_matmul(&a, &b, m, k, n);
        for j in 0..n {
            for i in 0..m {
                r[i * n + j] += bias[j];
            }
        }
        assert_eq!(c, r);
    }

    #[test]
    fn split_batch_cover_all() {
        for batch in 0..17 {
            for nt in 0..5 {
                let mut seen = vec![false; batch];
                for (s, e) in split_batch(batch, nt.max(1)) {
                    for i in s..e {
                        seen[i] = true;
                    }
                }
                assert!(seen.iter().all(|&v| v), "batch={batch} nt={nt}");
            }
        }
    }
}
