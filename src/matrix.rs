//! Generic dense matrices `Matrix<T>`.
//!
//! Row-major storage over a generic component type. Ring operations
//! (add/sub/mul/transpose/scalar) need only `+ - *` on `T`, so they work for any
//! of the crate's numeric types. Exact linear algebra is provided for two
//! concrete component types:
//!
//! - [`Matrix<Int>`](Matrix): a fraction-free (Bareiss) integer determinant.
//! - [`Matrix<Rational>`](Matrix): determinant, inverse, linear solve, and rank
//!   by exact Gaussian elimination over the rationals.
//!
//! For an *arbitrary* [`Field`] component — the finite
//! fields `GF(p)` / `GF(pᵏ)`, `Float`, … — the same operations are provided
//! generically by the [`FieldMatrix`] trait (Gaussian elimination with
//! pivoting). Bring it into scope with `use puremp::FieldMatrix;`.

use crate::ring::{Field, Ring};
use alloc::vec::Vec;
use core::fmt;

/// Dimension crossover below which [`Matrix::mul`] uses the naive triple loop
/// instead of recursing with Strassen–Winograd. A block product is taken by
/// Strassen only while every dimension of the (possibly recursively split)
/// operands strictly exceeds this; the recursion bottoms out in the naive
/// multiply at or below it.
const STRASSEN_THRESHOLD: usize = 24;

/// Minimum entry size (a [`Ring::multiply_cost_hint`], i.e. bit length for
/// `Int`/`Rational`) below which [`Matrix::mul`] stays on the naive path
/// regardless of dimension.
///
/// Strassen–Winograd replaces one of every eight element multiplies with a
/// bundle of extra element additions and block allocations. Measured on this
/// crate's `Int`/`Rational`, that is a net win only once a single element
/// multiply is dear enough — around a thousand bits. Below this cutoff (e.g.
/// machine-word-sized entries) the extra additions dominate and Strassen is
/// ~10–20% *slower* at every dimension, so the naive path is kept. Above it,
/// and above [`STRASSEN_THRESHOLD`] in dimension, Strassen wins: measured
/// ≈1.03–1.26× for `Int` (2000-bit entries, `n` = 32…64) and ≈1.25–1.53× for
/// large reduced `Rational`s, both growing with entry size and dimension.
const STRASSEN_MIN_ENTRY_BITS: u64 = 1024;

/// A dense `rows × cols` matrix stored row-major.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Matrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T> Matrix<T> {
    /// Builds a `rows × cols` matrix from row-major data. Panics on a length
    /// mismatch.
    pub fn new(rows: usize, cols: usize, data: Vec<T>) -> Matrix<T> {
        assert_eq!(rows * cols, data.len(), "Matrix::new: data length mismatch");
        Matrix { rows, cols, data }
    }

    /// Builds a matrix from a list of rows. Panics if the rows differ in length.
    pub fn from_rows(rows: Vec<Vec<T>>) -> Matrix<T> {
        let r = rows.len();
        let c = rows.first().map_or(0, |row| row.len());
        let mut data = Vec::with_capacity(r * c);
        for row in rows {
            assert_eq!(row.len(), c, "Matrix::from_rows: ragged rows");
            data.extend(row);
        }
        Matrix {
            rows: r,
            cols: c,
            data,
        }
    }

    /// Returns the number of rows.
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns.
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Returns `true` if the matrix is square.
    #[inline]
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// Returns the entry at `(row, col)`.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> &T {
        &self.data[row * self.cols + col]
    }

    /// Sets the entry at `(row, col)`.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        self.data[row * self.cols + col] = value;
    }

    /// Returns the entries in row-major order.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }
}

impl<T: Clone> Matrix<T> {
    /// Builds a `rows × cols` matrix with every entry a clone of `value`.
    ///
    /// This is the context-carrying constructor: unlike [`zeros`](Matrix::zeros)
    /// (which needs a context-free `Default`), it works for rings whose zero
    /// depends on a runtime context (`ModInt`, `GfElement`) — pass their
    /// [`Ring::zero`].
    pub fn filled(value: T, rows: usize, cols: usize) -> Matrix<T> {
        Matrix {
            rows,
            cols,
            data: alloc::vec![value; rows * cols],
        }
    }

    /// Returns the transpose.
    pub fn transpose(&self) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..self.rows {
            for j in 0..self.cols {
                data[j * self.rows + i] = self.data[i * self.cols + j].clone();
            }
        }
        Matrix {
            rows: self.cols,
            cols: self.rows,
            data,
        }
    }
}

impl<T: Clone + Default> Matrix<T> {
    /// Builds a `rows × cols` zero matrix.
    pub fn zeros(rows: usize, cols: usize) -> Matrix<T> {
        Matrix {
            rows,
            cols,
            data: alloc::vec![T::default(); rows * cols],
        }
    }
}

impl<T: Ring> Matrix<T> {
    /// Builds a `rows × cols` zero matrix, taking the ring's zero from `sample`.
    ///
    /// The context-carrying counterpart of [`zeros`](Matrix::zeros): use it for
    /// component rings whose zero depends on a runtime context (`ModInt`,
    /// `GfElement`).
    pub fn zeros_like(sample: &T, rows: usize, cols: usize) -> Matrix<T> {
        Matrix::filled(sample.zero(), rows, cols)
    }

    /// Builds the `n × n` identity, taking the ring's zero/one from `sample`.
    ///
    /// The context-carrying counterpart of the concrete `Matrix::<Int>::identity`
    /// / `Matrix::<Rational>::identity`.
    pub fn identity_like(sample: &T, n: usize) -> Matrix<T> {
        let mut m = Matrix::zeros_like(sample, n, n);
        let one = sample.one();
        for i in 0..n {
            m.set(i, i, one.clone());
        }
        m
    }
}

impl<T: Ring> Matrix<T> {
    /// Returns `self + rhs`. Panics on a shape mismatch.
    pub fn add(&self, rhs: &Matrix<T>) -> Matrix<T> {
        assert!(
            self.rows == rhs.rows && self.cols == rhs.cols,
            "Matrix::add: shape mismatch"
        );
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .iter()
                .zip(&rhs.data)
                .map(|(a, b)| a.clone() + b.clone())
                .collect(),
        }
    }

    /// Returns `self - rhs`. Panics on a shape mismatch.
    pub fn sub(&self, rhs: &Matrix<T>) -> Matrix<T> {
        assert!(
            self.rows == rhs.rows && self.cols == rhs.cols,
            "Matrix::sub: shape mismatch"
        );
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .iter()
                .zip(&rhs.data)
                .map(|(a, b)| a.clone() - b.clone())
                .collect(),
        }
    }

    /// Returns the matrix product `self · rhs`. Panics if the inner dimensions
    /// disagree.
    ///
    /// For component rings whose arithmetic is exact and associative
    /// ([`Ring::REASSOCIATIVE`]) the product is computed by the recursive
    /// **Strassen–Winograd** algorithm above a size threshold (7 recursive block
    /// products instead of 8), falling back to the naive triple loop below the
    /// threshold and for small or awkward shapes. The visible result is always
    /// **bit-identical** to the naive product.
    pub fn mul(&self, rhs: &Matrix<T>) -> Matrix<T> {
        assert_eq!(self.cols, rhs.rows, "Matrix::mul: inner dimension mismatch");
        // Fast path: only for exact rings (Strassen re-associates arithmetic, so
        // it is bit-identical to naive *only* when `+`/`−`/`×` are exact), only
        // once every dimension is comfortably above the crossover point, and only
        // when the entries are large enough that saving a multiply outweighs the
        // extra additions (sampled cheaply on one entry; both operands are
        // non-empty here).
        if T::REASSOCIATIVE
            && self.rows.min(self.cols).min(rhs.cols) > STRASSEN_THRESHOLD
            && self.data[0].multiply_cost_hint() >= STRASSEN_MIN_ENTRY_BITS
        {
            return self.strassen_mul(rhs);
        }
        self.naive_mul(rhs)
    }

    /// The classical `O(n³)` triple-loop product — the reference implementation
    /// and the base case of [`strassen_mul`](Matrix::strassen_mul).
    fn naive_mul(&self, rhs: &Matrix<T>) -> Matrix<T> {
        let out_len = self.rows * rhs.cols;
        // Accumulator zeros come from a sample cell (the ring's zero). The only
        // way both operands are empty yet a cell is needed is the degenerate
        // `m×0 · 0×n` product, whose ring cannot be inferred.
        let data: Vec<T> = match self.data.first().or_else(|| rhs.data.first()) {
            Some(sample) => alloc::vec![sample.zero(); out_len],
            None => {
                assert_eq!(
                    out_len, 0,
                    "Matrix::mul: cannot infer the ring's zero from empty operands"
                );
                Vec::new()
            }
        };
        let mut out = Matrix {
            rows: self.rows,
            cols: rhs.cols,
            data,
        };
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self.data[i * self.cols + k].clone();
                for j in 0..rhs.cols {
                    let prod = a.clone() * rhs.data[k * rhs.cols + j].clone();
                    let slot = &mut out.data[i * rhs.cols + j];
                    *slot = slot.clone() + prod;
                }
            }
        }
        out
    }

    /// Recursive Strassen–Winograd product. Assumes the inner dimension already
    /// matches (`self.cols == rhs.rows`); dispatched from [`mul`](Matrix::mul)
    /// only for exact rings above [`STRASSEN_THRESHOLD`].
    fn strassen_mul(&self, rhs: &Matrix<T>) -> Matrix<T> {
        let m = self.rows;
        let k = self.cols;
        let n = rhs.cols;
        // Base case: recurse down to the naive triple loop.
        if m.min(k).min(n) <= STRASSEN_THRESHOLD {
            return self.naive_mul(rhs);
        }
        // A sample entry gives the ring's zero for the padding cells. Both
        // operands are non-empty here (all dimensions exceed the threshold).
        let zero = self.data[0].zero();

        // Pad each dimension up to even so the four quadrants are equal-sized.
        let me = m + (m & 1);
        let ke = k + (k & 1);
        let ne = n + (n & 1);
        let hm = me / 2;
        let hk = ke / 2;
        let hn = ne / 2;

        let a = self.padded(me, ke, &zero);
        let b = rhs.padded(ke, ne, &zero);

        // A blocks (hm×hk) and B blocks (hk×hn).
        let a11 = a.block(0, 0, hm, hk);
        let a12 = a.block(0, hk, hm, hk);
        let a21 = a.block(hm, 0, hm, hk);
        let a22 = a.block(hm, hk, hm, hk);
        let b11 = b.block(0, 0, hk, hn);
        let b12 = b.block(0, hn, hk, hn);
        let b21 = b.block(hk, 0, hk, hn);
        let b22 = b.block(hk, hn, hk, hn);

        // Winograd's schedule (7 products, 15 additions; the addition-minimizing
        // variant of Strassen, *Numer. Math.* 13 (1969) / MCA §12.1). Verified by
        // symbolic expansion to equal the four naive block sums.
        let s1 = a21.add(&a22);
        let s2 = s1.sub(&a11);
        let s3 = a11.sub(&a21);
        let s4 = a12.sub(&s2);
        let s5 = b12.sub(&b11);
        let s6 = b22.sub(&s5);
        let s7 = b22.sub(&b12);
        let s8 = s6.sub(&b21);

        let p1 = s2.strassen_mul(&s6);
        let p2 = a11.strassen_mul(&b11);
        let p3 = a12.strassen_mul(&b21);
        let p4 = s3.strassen_mul(&s7);
        let p5 = s1.strassen_mul(&s5);
        let p6 = s4.strassen_mul(&b22);
        let p7 = a22.strassen_mul(&s8);

        let u1 = p1.add(&p2);
        let u2 = u1.add(&p4);
        let u3 = u1.add(&p5);

        let c11 = p2.add(&p3);
        let c12 = u3.add(&p6);
        let c21 = u2.sub(&p7);
        let c22 = u2.add(&p5);

        // Reassemble into the visible m×n result (dropping the padded row/column).
        let mut data = alloc::vec![zero; m * n];
        for i in 0..m {
            let (bi, ii) = if i < hm { (false, i) } else { (true, i - hm) };
            for j in 0..n {
                let (bj, jj) = if j < hn { (false, j) } else { (true, j - hn) };
                let cell = match (bi, bj) {
                    (false, false) => c11.get(ii, jj),
                    (false, true) => c12.get(ii, jj),
                    (true, false) => c21.get(ii, jj),
                    (true, true) => c22.get(ii, jj),
                };
                data[i * n + j] = cell.clone();
            }
        }
        Matrix {
            rows: m,
            cols: n,
            data,
        }
    }

    /// Returns a `rows × cols` copy with `self` in the top-left corner and `zero`
    /// elsewhere (`rows ≥ self.rows`, `cols ≥ self.cols`).
    fn padded(&self, rows: usize, cols: usize, zero: &T) -> Matrix<T> {
        let mut data = alloc::vec![zero.clone(); rows * cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                data[i * cols + j] = self.data[i * self.cols + j].clone();
            }
        }
        Matrix { rows, cols, data }
    }

    /// Extracts the `hr × hc` sub-block whose top-left corner is `(r0, c0)`.
    fn block(&self, r0: usize, c0: usize, hr: usize, hc: usize) -> Matrix<T> {
        let mut data = Vec::with_capacity(hr * hc);
        for i in 0..hr {
            for j in 0..hc {
                data.push(self.data[(r0 + i) * self.cols + (c0 + j)].clone());
            }
        }
        Matrix {
            rows: hr,
            cols: hc,
            data,
        }
    }

    /// Returns `self · scalar`.
    pub fn scalar_mul(&self, scalar: &T) -> Matrix<T> {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .iter()
                .map(|a| a.clone() * scalar.clone())
                .collect(),
        }
    }
}

impl<T: Ring> Matrix<T> {
    /// Returns `-self`.
    pub fn neg(&self) -> Matrix<T> {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|a| -a.clone()).collect(),
        }
    }
}

impl<T: fmt::Display> fmt::Display for Matrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.rows {
            f.write_str("[")?;
            for j in 0..self.cols {
                if j > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{}", self.get(i, j))?;
            }
            f.write_str("]")?;
            if i + 1 < self.rows {
                f.write_str("\n")?;
            }
        }
        Ok(())
    }
}

// ---- exact integer determinant (Bareiss) ----

#[cfg(feature = "int")]
impl Matrix<crate::int::Int> {
    /// Returns the `n × n` integer identity matrix.
    pub fn identity(n: usize) -> Matrix<crate::int::Int> {
        use crate::int::Int;
        let mut m = Matrix::zeros(n, n);
        for i in 0..n {
            m.set(i, i, Int::ONE);
        }
        m
    }

    /// Returns the exact determinant via the fraction-free Bareiss algorithm.
    /// Panics if the matrix is not square.
    pub fn determinant(&self) -> crate::int::Int {
        use crate::int::Int;
        assert!(self.is_square(), "determinant: matrix must be square");
        let n = self.rows;
        if n == 0 {
            return Int::ONE;
        }
        let idx = |i: usize, j: usize| i * n + j;
        let mut a = self.data.clone();
        let mut prev = Int::ONE;
        let mut sign = false; // true == negated
        for k in 0..n - 1 {
            if a[idx(k, k)].is_zero() {
                match (k + 1..n).find(|&r| !a[idx(r, k)].is_zero()) {
                    Some(r) => {
                        for c in 0..n {
                            a.swap(idx(k, c), idx(r, c));
                        }
                        sign = !sign;
                    }
                    None => return Int::ZERO,
                }
            }
            for i in k + 1..n {
                for j in k + 1..n {
                    let num = Int::sub(
                        &Int::mul(&a[idx(i, j)], &a[idx(k, k)]),
                        &Int::mul(&a[idx(i, k)], &a[idx(k, j)]),
                    );
                    a[idx(i, j)] = Int::div_exact(&num, &prev); // exact by Bareiss
                }
            }
            prev = a[idx(k, k)].clone();
        }
        let det = a[idx(n - 1, n - 1)].clone();
        if sign { det.neg() } else { det }
    }
}

// ---- exact integer Hermite / Smith normal forms ----
//
// Clean-room implementation from the open literature: H. Cohen, *A Course in
// Computational Algebraic Number Theory*, §2.4.2–2.4.4 (Hermite normal form)
// and §2.4.4 / algorithm 2.4.14 (Smith normal form). Only the mathematical
// ideas — unimodular row/column operations, extended-gcd pivoting, reduction of
// off-pivot entries modulo the pivot, and the alternating row/column clearing
// with a divisibility repair step for Smith form — are used; no third-party code
// was consulted.

/// Applies the unimodular 2×2 row operation with coefficient matrix
/// `[[c0, c1], [c2, c3]]` to rows `ra`, `rb` of a `width`-column row-major matrix:
/// `row_ra ← c0·row_ra + c1·row_rb`, `row_rb ← c2·row_ra + c3·row_rb`.
#[cfg(feature = "int")]
fn mat_row_op(
    data: &mut [crate::int::Int],
    width: usize,
    ra: usize,
    rb: usize,
    c: &[crate::int::Int; 4],
) {
    for k in 0..width {
        let ia = ra * width + k;
        let ib = rb * width + k;
        let x = data[ia].clone();
        let y = data[ib].clone();
        data[ia] = c[0].mul(&x).add(&c[1].mul(&y));
        data[ib] = c[2].mul(&x).add(&c[3].mul(&y));
    }
}

/// The column analogue of [`mat_row_op`] over a matrix with `cols` columns and
/// `nrows` rows, combining columns `ca`, `cb`.
#[cfg(feature = "int")]
fn mat_col_op(
    data: &mut [crate::int::Int],
    cols: usize,
    nrows: usize,
    ca: usize,
    cb: usize,
    c: &[crate::int::Int; 4],
) {
    for r in 0..nrows {
        let ia = r * cols + ca;
        let ib = r * cols + cb;
        let x = data[ia].clone();
        let y = data[ib].clone();
        data[ia] = c[0].mul(&x).add(&c[1].mul(&y));
        data[ib] = c[2].mul(&x).add(&c[3].mul(&y));
    }
}

/// Swaps rows `ra`, `rb` of a `width`-column row-major matrix (a unimodular
/// operation of determinant −1).
#[cfg(feature = "int")]
fn mat_swap_rows(data: &mut [crate::int::Int], width: usize, ra: usize, rb: usize) {
    for k in 0..width {
        data.swap(ra * width + k, rb * width + k);
    }
}

/// Swaps columns `ca`, `cb` of a matrix with `cols` columns and `nrows` rows.
#[cfg(feature = "int")]
fn mat_swap_cols(data: &mut [crate::int::Int], cols: usize, nrows: usize, ca: usize, cb: usize) {
    for r in 0..nrows {
        data.swap(r * cols + ca, r * cols + cb);
    }
}

/// Negates row `ra` (a unimodular operation of determinant −1).
#[cfg(feature = "int")]
fn mat_negate_row(data: &mut [crate::int::Int], width: usize, ra: usize) {
    for k in 0..width {
        let i = ra * width + k;
        data[i] = data[i].neg();
    }
}

/// Coefficients of the unimodular 2×2 operation that clears the pivot-line entry
/// `b` (in the row/column carrying the Smith pivot `a`, `a ≠ 0`).
///
/// The subtlety that makes Smith reduction terminate: when `a | b`, the entry is
/// removed by a plain *subtraction of a multiple* (`[[1,0],[-b/a,1]]`), which
/// leaves the **pivot line unchanged** — so it never re-dirties the orthogonal
/// line already cleared. Only when `a ∤ b` do we take the general extended-gcd
/// combination `[[x,y],[-b/g,a/g]]` (`g = gcd(a,b)`), which replaces the pivot by
/// `g < |a|`; because the pivot strictly shrinks on every such step and is a
/// positive integer, the alternating row/column clearing converges. (A naive
/// unconditional extended-gcd step can, e.g. for `a = b`, return a Bézout pair
/// that merely *permutes* the pivot line and loops forever.)
#[cfg(feature = "int")]
fn snf_clear_coeffs(a: &crate::int::Int, b: &crate::int::Int) -> [crate::int::Int; 4] {
    use crate::int::Int;
    let (q, r) = b.div_rem_trunc(a);
    if r.is_zero() {
        // b = q·a: row_i ← row_i − q·row_pivot (pivot line untouched).
        [Int::ONE, Int::ZERO, q.neg(), Int::ONE]
    } else {
        let (g, x, y) = a.extended_gcd(b); // g = a·x + b·y, 0 ≤ g < |a|
        let ag = a.div_exact(&g);
        let bg = b.div_exact(&g).neg();
        [x, y, bg, ag]
    }
}

/// Row Hermite normal form of an `n × n` **nonsingular** integer matrix computed
/// with *modulo-determinant arithmetic* (Domich–Kannan–Trotter 1987; Cohen,
/// *CCANT* §2.4.2). `src` is the row-major matrix and `det_abs = |det(src)| > 0`.
///
/// The row lattice `L = rowspan(src)` has covolume `D = |det|`. After the first
/// `col` pivots `p₀,…,p_{col-1}` are fixed, the lattice projected onto the
/// remaining coordinates has covolume `R = D / (p₀···p_{col-1})`, so it contains
/// `R·ℤ^{n-col}` — i.e. `R·e_j` is a lattice vector for every remaining column
/// `j`. Reducing every entry modulo this **shrinking modulus `R`** (subtracting
/// multiples of those `R·e_j`) keeps all intermediates in `[0, R) ⊆ [0, D)`,
/// killing the coefficient explosion of the plain extended-gcd reduction.
///
/// The pivot for column `col` is `gcd(a, R)` where `a` is the gcd of the sub-column
/// entries (the `R` folding in the modulus generator `R·e_col`); it divides `R`,
/// and `R` then shrinks by it — so the diagonal telescopes to `∏pᵢ = D`. Every
/// pivot lies in `[1, R]` and every above-pivot entry is reduced into `[0, pivot)`,
/// giving the *same* canonical `H` as [`Matrix::hermite_normal_form`].
#[cfg(feature = "int")]
fn hnf_mod_det(
    src: &[crate::int::Int],
    n: usize,
    det_abs: &crate::int::Int,
) -> Vec<crate::int::Int> {
    use crate::int::Int;
    let idx = |i: usize, j: usize| i * n + j;
    let mut r = det_abs.clone(); // current modulus D / ∏(pivots so far)
    // Work with every entry reduced into [0, R).
    let mut w: Vec<Int> = src.iter().map(|e| e.rem_euclid(&r)).collect();
    for col in 0..n {
        let row = col; // nonsingular ⇒ the pivot for column `col` lands on row `col`
        // Combine the sub-column below the pivot into a single gcd entry at
        // (row, col) with unimodular row operations, keeping everything mod R.
        for i in row + 1..n {
            if w[idx(i, col)].is_zero() {
                continue;
            }
            let a = w[idx(row, col)].clone();
            let b = w[idx(i, col)].clone();
            // extended_gcd handles a == 0 (yields the swap-with-sign combination).
            let (g, x, y) = a.extended_gcd(&b);
            let ag = a.div_exact(&g);
            let bg = b.div_exact(&g).neg();
            for k in 0..n {
                let xr = w[idx(row, k)].clone();
                let yr = w[idx(i, k)].clone();
                w[idx(row, k)] = x.mul(&xr).add(&y.mul(&yr)).rem_euclid(&r);
                w[idx(i, k)] = bg.mul(&xr).add(&ag.mul(&yr)).rem_euclid(&r);
            }
        }
        // Fold in the modulus generator `R·e_col`: the pivot is `p = gcd(a, R)`.
        // With `p = a·x + R·y`, replacing the row by `x·row + y·(R·e_col)` sets
        // the pivot entry to `p` and scales the rest by `x` (mod R). When `a = 0`
        // we get `x = 0`, `p = R`: the row becomes the pure modulus vector.
        let a = w[idx(row, col)].clone();
        let (p, x, _y) = a.extended_gcd(&r);
        for k in 0..n {
            let v = w[idx(row, k)].clone();
            w[idx(row, k)] = x.mul(&v).rem_euclid(&r);
        }
        w[idx(row, col)] = p.clone(); // pivot in [1, R]
        // Reduce every entry above the pivot into `[0, p)`.
        for i in 0..row {
            let q = w[idx(i, col)].div_floor(&p); // remainder in [0, p)
            if q.is_zero() {
                continue;
            }
            for k in 0..n {
                let sub = q.mul(&w[idx(row, k)]);
                w[idx(i, k)] = w[idx(i, k)].sub(&sub).rem_euclid(&r);
            }
        }
        // Shrink the modulus by the pivot (exact: p | R) and re-reduce the rows
        // still to be processed into the new, smaller range.
        r = r.div_exact(&p);
        if !r.is_one() {
            for i in row + 1..n {
                for k in 0..n {
                    w[idx(i, k)] = w[idx(i, k)].rem_euclid(&r);
                }
            }
        }
    }
    w
}

/// Heuristic gate for the modulo-determinant HNF route. The modular path pays a
/// fixed cost (a determinant / fraction-free pass) but then keeps every entry
/// bounded by `|det|`, whereas the plain reduction blows intermediate entries up
/// super-linearly in both the dimension and the entry size.
///
/// Measurements (the `#[ignore]`d `bench_modular_vs_reference` test) put the
/// crossover around dimension 16: at `n = 8` the reference path is faster at every
/// entry size, at `n = 16` the modular path wins once entries exceed ~64 bits
/// (≈2×), and by `n = 24` it wins across the board (2.6× at 32-bit, 8–40× at
/// 128–512-bit). We therefore route only when the dimension is at least
/// `MIN_DIM`, and additionally require sizable entries (`MIN_BITS`) in the
/// `MIN_DIM..BIG_DIM` band where the win is entry-size-dependent; from `BIG_DIM`
/// up the modular path wins even for small entries.
#[cfg(feature = "int")]
fn hnf_modular_worthwhile(a: &Matrix<crate::int::Int>) -> bool {
    const MIN_DIM: usize = 12;
    const BIG_DIM: usize = 20;
    const MIN_BITS: u32 = 64;
    let n = a.rows;
    if n < MIN_DIM {
        return false;
    }
    if n >= BIG_DIM {
        return true;
    }
    let max_bits = a.data.iter().map(|e| e.bit_len()).max().unwrap_or(0);
    max_bits >= MIN_BITS
}

#[cfg(feature = "int")]
impl Matrix<crate::int::Int> {
    /// Core row-style Hermite-normal-form reduction.
    ///
    /// Returns `(H, U?, rank)`, where `H = U · self` is the row HNF and, when
    /// `want_u`, `U` is the accumulated unimodular transform (row-major `m × m`).
    /// `rank` is the number of nonzero rows of `H` (the ℚ-rank of `self`).
    fn hnf_impl(
        &self,
        want_u: bool,
    ) -> (Vec<crate::int::Int>, Option<Vec<crate::int::Int>>, usize) {
        use crate::int::Int;
        let m = self.rows;
        let n = self.cols;
        let mut h = self.data.clone();
        let mut u = if want_u {
            Matrix::<Int>::identity(m).data
        } else {
            Vec::new()
        };
        let mut row = 0usize;
        for col in 0..n {
            if row >= m {
                break;
            }
            // Reduce the sub-column below the pivot to a single gcd entry at
            // `row` using unimodular row combinations (extended-gcd pivoting).
            for i in row + 1..m {
                let b = h[i * n + col].clone();
                if b.is_zero() {
                    continue;
                }
                let a = h[row * n + col].clone();
                let (g, x, y) = a.extended_gcd(&b); // g = a·x + b·y ≥ 0
                let ag = a.div_exact(&g);
                let bg = b.div_exact(&g).neg();
                // [[x, y], [-b/g, a/g]] has determinant (a·x + b·y)/g = 1.
                let coeffs = [x, y, bg, ag];
                mat_row_op(&mut h, n, row, i, &coeffs);
                if want_u {
                    mat_row_op(&mut u, m, row, i, &coeffs);
                }
            }
            if h[row * n + col].is_zero() {
                continue; // wholly-zero sub-column ⇒ not a pivot column
            }
            // Normalize the pivot to be positive.
            if h[row * n + col].is_negative() {
                mat_negate_row(&mut h, n, row);
                if want_u {
                    mat_negate_row(&mut u, m, row);
                }
            }
            let piv = h[row * n + col].clone();
            // Reduce every entry *above* the pivot into `[0, piv)`.
            for i in 0..row {
                let q = h[i * n + col].div_floor(&piv); // remainder in [0, piv)
                if q.is_zero() {
                    continue;
                }
                // row_i ← row_i − q·row_{row}.
                let coeffs = [Int::ONE, q.neg(), Int::ZERO, Int::ONE];
                mat_row_op(&mut h, n, i, row, &coeffs);
                if want_u {
                    mat_row_op(&mut u, m, i, row, &coeffs);
                }
            }
            row += 1;
        }
        (h, want_u.then_some(u), row)
    }

    /// Returns the **row Hermite normal form** `H` of `self`.
    ///
    /// `H` is the unique matrix row-equivalent to `self` (i.e. `H = U · self` for
    /// some unimodular `U`) in *row echelon form* with:
    ///
    /// - pivot columns `j₀ < j₁ < … < j_{r−1}` (`r` = rank), row `i` having its
    ///   leading nonzero (the pivot) at column `jᵢ`;
    /// - every pivot **positive**;
    /// - every entry **above** a pivot reduced into `[0, pivot)`;
    /// - the trailing `m − r` rows zero.
    ///
    /// Because it is canonical, two matrices with the same row lattice (the same
    /// ℤ-span of rows) have the *same* HNF. For a square nonsingular matrix the
    /// product of the diagonal equals `|det|`.
    ///
    /// Off-pivot (above-pivot) entries are reduced modulo the pivot eagerly, which
    /// keeps `H`'s entries bounded by the pivots; pivots are produced by
    /// extended-gcd combination (so they are gcds, not growing products).
    pub fn hermite_normal_form(&self) -> Matrix<crate::int::Int> {
        if let Some(h) = self.hnf_modular() {
            return Matrix::new(self.rows, self.cols, h);
        }
        let (h, _, _) = self.hnf_impl(false);
        Matrix::new(self.rows, self.cols, h)
    }

    /// Row HNF via *modulo-determinant arithmetic* ([`hnf_mod_det`]), returning
    /// `Some(H)` only for full-rank inputs above the size/entry threshold where
    /// the modular route beats the plain reduction, and `None` otherwise (small
    /// inputs, or rank-deficient inputs, which fall back to [`Self::hnf_impl`]).
    ///
    /// The exact `H` is identical to [`Self::hermite_normal_form`]'s reference
    /// path; only intermediate coefficient growth differs.
    fn hnf_modular(&self) -> Option<Vec<crate::int::Int>> {
        let m = self.rows;
        let n = self.cols;
        if m == 0 || n == 0 || m > n {
            return None; // handled by the reference path (incl. tall matrices)
        }
        if !hnf_modular_worthwhile(self) {
            return None;
        }
        if m == n {
            // Square: nonsingular ⇔ full rank. |det| is the lattice determinant.
            let det = self.determinant();
            if det.is_zero() {
                return None; // rank-deficient ⇒ reference path
            }
            return Some(hnf_mod_det(&self.data, n, &det.abs()));
        }
        // Wide (m < n): full row rank only. Find the pivot columns and the
        // determinant of that square minor, HNF the minor modularly, then fill
        // the remaining columns from the (exact) unimodular transform. The exact
        // rational solve for the non-pivot columns carries a real overhead, so
        // the wide crossover sits higher than the square one (measurements: it
        // only wins reliably from ~dimension 20); gate it accordingly. The exact
        // solve needs the `rational` layer — without it, wide inputs fall back to
        // the reference path.
        #[cfg(feature = "rational")]
        {
            const WIDE_MIN_DIM: usize = 20;
            if m >= WIDE_MIN_DIM {
                return self.hnf_modular_wide();
            }
        }
        None
    }

    /// Wide-matrix (`m < n`) branch of [`Self::hnf_modular`], for full row rank.
    ///
    /// The HNF pivots sit at the leftmost independent columns `P` (found by a
    /// fraction-free pass). Restricted to `P`, the HNF equals the HNF of the
    /// nonsingular minor `A_P` — computed modularly — because the row lattice
    /// projected onto `P` is `rowspan(A_P)`. The remaining columns are
    /// `H_R = U·A_R` where `U = H_P·A_P⁻¹` is the (unique) transform of the
    /// square subproblem; equivalently `H_R = H_P·(A_P⁻¹·A_R)`, which we obtain
    /// from an exact rational solve. Returns `None` if not full row rank (⇒
    /// reference path).
    #[cfg(feature = "rational")]
    fn hnf_modular_wide(&self) -> Option<Vec<crate::int::Int>> {
        use crate::int::Int;
        use crate::rational::Rational;
        let m = self.rows;
        let n = self.cols;
        let (piv, det_abs) = self.pivot_columns_and_det()?;
        if piv.len() != m {
            return None; // not full row rank ⇒ reference path
        }
        let is_pivot = {
            let mut v = alloc::vec![false; n];
            for &c in &piv {
                v[c] = true;
            }
            v
        };
        let rest: Vec<usize> = (0..n).filter(|&c| !is_pivot[c]).collect();
        // Square minor A_P (m × m) and its modular HNF H_P.
        let a_p: Vec<Int> = (0..m)
            .flat_map(|i| piv.iter().map(move |&c| self.data[i * n + c].clone()))
            .collect();
        let h_p = hnf_mod_det(&a_p, m, &det_abs);
        // Exact rational solve A_P · Z = A_R  (Z = A_P⁻¹·A_R), then H_R = H_P·Z.
        let a_p_rat = Matrix::new(
            m,
            m,
            a_p.iter()
                .map(|e| Rational::from_integer(e.clone()))
                .collect(),
        );
        let inv = a_p_rat.inverse()?; // nonsingular ⇒ Some
        let h_out = {
            // z[i][t] = (A_P⁻¹ · A_R)[i][t]
            let ncols = rest.len();
            let mut z = alloc::vec![Rational::ZERO; m * ncols];
            for (t, &c) in rest.iter().enumerate() {
                for i in 0..m {
                    let mut acc = Rational::ZERO;
                    for k in 0..m {
                        let a_rk = Rational::from_integer(self.data[k * n + c].clone());
                        acc = Rational::add(&acc, &Rational::mul(inv.get(i, k), &a_rk));
                    }
                    z[i * ncols + t] = acc;
                }
            }
            // H_R = H_P · Z (integer); assemble full H at original column order.
            let mut h = alloc::vec![Int::ZERO; m * n];
            for i in 0..m {
                for (k, &c) in piv.iter().enumerate() {
                    h[i * n + c] = h_p[i * m + k].clone();
                }
                for (t, &c) in rest.iter().enumerate() {
                    let mut acc = Rational::ZERO;
                    for k in 0..m {
                        let hpk = Rational::from_integer(h_p[i * m + k].clone());
                        acc = Rational::add(&acc, &Rational::mul(&hpk, &z[k * ncols + t]));
                    }
                    debug_assert!(acc.is_integer(), "H_R entry not integral");
                    h[i * n + c] = acc.numerator().clone();
                }
            }
            h
        };
        Some(h_out)
    }

    /// Fraction-free (Bareiss) forward elimination that returns the leftmost
    /// independent columns (the HNF pivot columns) together with `|det|` of the
    /// square submatrix on those columns. All intermediate entries are integer
    /// minors (bounded by the determinant), so this pivot search never suffers
    /// the coefficient blow-up of the plain reduction.
    #[cfg(feature = "rational")]
    fn pivot_columns_and_det(&self) -> Option<(Vec<usize>, crate::int::Int)> {
        use crate::int::Int;
        let m = self.rows;
        let n = self.cols;
        let mut a = self.data.clone();
        let idx = |i: usize, j: usize| i * n + j;
        let mut prev = Int::ONE;
        let mut piv: Vec<usize> = Vec::with_capacity(m);
        let mut r = 0usize;
        for col in 0..n {
            if r == m {
                break;
            }
            let pr = (r..m).find(|&i| !a[idx(i, col)].is_zero());
            let pr = match pr {
                Some(p) => p,
                None => continue, // dependent column ⇒ no pivot here
            };
            if pr != r {
                for c in col..n {
                    a.swap(idx(r, c), idx(pr, c));
                }
            }
            piv.push(col);
            let arc = a[idx(r, col)].clone();
            for i in r + 1..m {
                let aic = a[idx(i, col)].clone();
                for j in col + 1..n {
                    let num = a[idx(i, j)].mul(&arc).sub(&aic.mul(&a[idx(r, j)]));
                    a[idx(i, j)] = num.div_exact(&prev); // exact by the Bareiss identity
                }
                a[idx(i, col)] = Int::ZERO;
            }
            prev = arc;
            r += 1;
        }
        Some((piv, prev.abs()))
    }

    /// Returns `(H, U)` where `H` is the [row HNF](Self::hermite_normal_form) and
    /// `U` is a unimodular matrix (`|det U| = 1`) with **`H = U · self`** (the
    /// transform acts on the **left**). `U` is `m × m` for an `m × n` input.
    pub fn hermite_normal_form_with_transform(
        &self,
    ) -> (Matrix<crate::int::Int>, Matrix<crate::int::Int>) {
        let (h, u, _) = self.hnf_impl(true);
        (
            Matrix::new(self.rows, self.cols, h),
            Matrix::new(self.rows, self.rows, u.expect("transform requested")),
        )
    }

    /// Core Smith-normal-form reduction.
    ///
    /// Returns `(D, rank, U?, V?)` with `D = U · self · V` diagonal,
    /// `diag(D) = (d₀, …, d_{r−1}, 0, …)` and `dᵢ | dᵢ₊₁`. `U` (`m × m`) and `V`
    /// (`n × n`) are the unimodular transforms, returned only when requested.
    #[allow(clippy::type_complexity)]
    fn snf_impl(
        &self,
        want_u: bool,
        want_v: bool,
    ) -> (
        Vec<crate::int::Int>,
        usize,
        Option<Vec<crate::int::Int>>,
        Option<Vec<crate::int::Int>>,
    ) {
        use crate::int::Int;
        let m = self.rows;
        let n = self.cols;
        let mut d = self.data.clone();
        let mut u = if want_u {
            Matrix::<Int>::identity(m).data
        } else {
            Vec::new()
        };
        let mut v = if want_v {
            Matrix::<Int>::identity(n).data
        } else {
            Vec::new()
        };
        let kmax = m.min(n);
        let mut t = 0usize;
        'main: while t < kmax {
            loop {
                // Ensure a nonzero pivot at (t, t); otherwise pull one in, or stop
                // when the whole trailing submatrix is zero.
                if d[t * n + t].is_zero() {
                    let mut found = None;
                    'search: for i in t..m {
                        for j in t..n {
                            if !d[i * n + j].is_zero() {
                                found = Some((i, j));
                                break 'search;
                            }
                        }
                    }
                    match found {
                        None => break 'main, // trailing submatrix all zero ⇒ done
                        Some((pi, pj)) => {
                            if pi != t {
                                mat_swap_rows(&mut d, n, t, pi);
                                if want_u {
                                    mat_swap_rows(&mut u, m, t, pi);
                                }
                            }
                            if pj != t {
                                mat_swap_cols(&mut d, n, m, t, pj);
                                if want_v {
                                    mat_swap_cols(&mut v, n, n, t, pj);
                                }
                            }
                        }
                    }
                }
                let mut changed = false;
                // Clear the pivot column below (t, t).
                for i in t + 1..m {
                    let b = d[i * n + t].clone();
                    if b.is_zero() {
                        continue;
                    }
                    let a = d[t * n + t].clone();
                    let coeffs = snf_clear_coeffs(&a, &b);
                    mat_row_op(&mut d, n, t, i, &coeffs);
                    if want_u {
                        mat_row_op(&mut u, m, t, i, &coeffs);
                    }
                    changed = true;
                }
                // Clear the pivot row right of (t, t) (may re-dirty the column).
                for j in t + 1..n {
                    let b = d[t * n + j].clone();
                    if b.is_zero() {
                        continue;
                    }
                    let a = d[t * n + t].clone();
                    let coeffs = snf_clear_coeffs(&a, &b);
                    mat_col_op(&mut d, n, m, t, j, &coeffs);
                    if want_v {
                        mat_col_op(&mut v, n, n, t, j, &coeffs);
                    }
                    changed = true;
                }
                if changed {
                    continue; // re-clear until the cross is a single pivot
                }
                // Divisibility repair: the pivot must divide the whole trailing
                // block. If not, fold an offending row into the pivot row and
                // re-reduce — this strictly shrinks the pivot (a gcd), so it
                // terminates.
                let piv = d[t * n + t].clone();
                let mut fixed = false;
                'divs: for i in t + 1..m {
                    for j in t + 1..n {
                        if !piv.divides(&d[i * n + j]) {
                            let coeffs = [Int::ONE, Int::ONE, Int::ZERO, Int::ONE]; // row_t += row_i
                            mat_row_op(&mut d, n, t, i, &coeffs);
                            if want_u {
                                mat_row_op(&mut u, m, t, i, &coeffs);
                            }
                            fixed = true;
                            break 'divs;
                        }
                    }
                }
                if fixed {
                    continue;
                }
                break;
            }
            // Normalize the invariant factor to be positive.
            if d[t * n + t].is_negative() {
                mat_negate_row(&mut d, n, t);
                if want_u {
                    mat_negate_row(&mut u, m, t);
                }
            }
            t += 1;
        }
        let rank = (0..kmax).filter(|&i| !d[i * n + i].is_zero()).count();
        (d, rank, want_u.then_some(u), want_v.then_some(v))
    }

    /// Returns the **Smith normal form** `D` of `self`: the `m × n` diagonal
    /// matrix `diag(d₀, …, d_{r−1}, 0, …)` with each `dᵢ > 0` and
    /// `dᵢ | dᵢ₊₁` (the invariant factors), such that `D = U · self · V` for some
    /// unimodular `U`, `V`.
    ///
    /// For a square nonsingular matrix `∏ dᵢ = |det|`; in general `d₀` is the gcd
    /// of all entries and `d₀·d₁·…·d_{k−1}` is the gcd of all `k × k` minors.
    pub fn smith_normal_form(&self) -> Matrix<crate::int::Int> {
        let (d, _, _, _) = self.snf_impl(false, false);
        Matrix::new(self.rows, self.cols, d)
    }

    /// Returns `(U, D, V)` where `D` is the [Smith normal form](Self::smith_normal_form)
    /// and `U` (`m × m`), `V` (`n × n`) are unimodular with **`D = U · self · V`**.
    pub fn smith_normal_form_with_transforms(
        &self,
    ) -> (
        Matrix<crate::int::Int>,
        Matrix<crate::int::Int>,
        Matrix<crate::int::Int>,
    ) {
        let (d, _, u, v) = self.snf_impl(true, true);
        (
            Matrix::new(self.rows, self.rows, u.expect("transform requested")),
            Matrix::new(self.rows, self.cols, d),
            Matrix::new(self.cols, self.cols, v.expect("transform requested")),
        )
    }

    /// Returns the **rank over ℚ** — the number of linearly independent rows,
    /// equal to the number of nonzero rows of the [row HNF](Self::hermite_normal_form).
    pub fn rank(&self) -> usize {
        self.hnf_impl(false).2
    }

    /// Returns the **invariant factors** `d₀ | d₁ | … | d_{r−1}` (the nonzero
    /// Smith-normal-form diagonal, each `> 0`).
    ///
    /// Viewing `self` as the matrix of a map `ℤⁿ → ℤᵐ` (columns are the images of
    /// the standard basis), the cokernel `ℤᵐ / (self·ℤⁿ)` is isomorphic to
    /// `⨁ᵢ ℤ/dᵢℤ ⊕ ℤ^{m−r}` — its torsion part is described by these factors
    /// (dropping any `dᵢ = 1`) and its free rank is `m − r`, where `r` is the
    /// [rank](Self::rank).
    pub fn invariant_factors(&self) -> Vec<crate::int::Int> {
        let (d, rank, _, _) = self.snf_impl(false, false);
        let n = self.cols;
        (0..rank).map(|i| d[i * n + i].clone()).collect()
    }

    /// Returns a ℤ-basis of the **integer kernel** `{x ∈ ℤⁿ : self·x = 0}`, as a
    /// list of `n`-long column vectors (empty when the kernel is trivial).
    ///
    /// From `D = U·self·V` (Smith form): `self·x = 0 ⇔ D·(V⁻¹x) = 0`, whose
    /// solution space is spanned by the standard basis vectors past the rank, so
    /// the last `n − r` columns of `V` form the basis.
    pub fn kernel(&self) -> Vec<Vec<crate::int::Int>> {
        let n = self.cols;
        let (_, rank, _, v) = self.snf_impl(false, true);
        let v = v.expect("transform requested");
        (rank..n)
            .map(|j| (0..n).map(|i| v[i * n + j].clone()).collect())
            .collect()
    }

    /// Returns a ℤ-basis of the **integer image** (column lattice)
    /// `self·ℤⁿ ⊆ ℤᵐ`, as a list of `m`-long column vectors (empty when `self`
    /// is zero).
    ///
    /// Computed as the nonzero rows of the row HNF of `selfᵀ` (a basis of the
    /// lattice spanned by the columns of `self`).
    pub fn image_basis(&self) -> Vec<Vec<crate::int::Int>> {
        let m = self.rows;
        let (h, _, rank) = self.transpose().hnf_impl(false);
        // selfᵀ is n × m; its HNF has `rank` nonzero rows, each of length m.
        (0..rank)
            .map(|i| (0..m).map(|k| h[i * m + k].clone()).collect())
            .collect()
    }

    /// Solves `self · x = b` over the integers, returning some `x ∈ ℤⁿ`, or
    /// `None` when no integer solution exists.
    ///
    /// Uses the Smith form `D = U·self·V`: with `c = U·b` and `y = V⁻¹x` the
    /// system becomes `D·y = c`, which is solvable iff `dᵢ | cᵢ` for `i < r` and
    /// `cᵢ = 0` for `i ≥ r`; then `x = V·y`. Panics if `b`'s length is not the row
    /// count. When the kernel is nontrivial the solution is one particular
    /// solution (add any [`kernel`](Self::kernel) vector for others).
    pub fn solve_integer(&self, b: &[crate::int::Int]) -> Option<Vec<crate::int::Int>> {
        use crate::int::Int;
        let m = self.rows;
        let n = self.cols;
        assert_eq!(b.len(), m, "solve_integer: right-hand side length mismatch");
        let (d, rank, u, v) = self.snf_impl(true, true);
        let u = u.expect("transform requested");
        let v = v.expect("transform requested");
        // c = U · b.
        let mut c = alloc::vec![Int::ZERO; m];
        for (i, ci) in c.iter_mut().enumerate() {
            let mut acc = Int::ZERO;
            for (k, bk) in b.iter().enumerate() {
                acc = acc.add(&u[i * m + k].mul(bk));
            }
            *ci = acc;
        }
        // Solve D · y = c.
        let mut y = alloc::vec![Int::ZERO; n];
        for i in 0..m {
            if i < rank {
                let di = &d[i * n + i];
                let (q, r) = c[i].div_rem_trunc(di);
                if !r.is_zero() {
                    return None; // dᵢ ∤ cᵢ
                }
                y[i] = q;
            } else if !c[i].is_zero() {
                return None; // inconsistent
            }
        }
        // x = V · y.
        let mut x = alloc::vec![Int::ZERO; n];
        for (i, xi) in x.iter_mut().enumerate() {
            let mut acc = Int::ZERO;
            for (k, yk) in y.iter().enumerate() {
                acc = acc.add(&v[i * n + k].mul(yk));
            }
            *xi = acc;
        }
        Some(x)
    }
}

// ---- exact rational linear algebra ----

#[cfg(feature = "rational")]
impl Matrix<crate::rational::Rational> {
    /// Returns the `n × n` rational identity matrix.
    pub fn identity(n: usize) -> Matrix<crate::rational::Rational> {
        use crate::rational::Rational;
        let mut m = Matrix::zeros(n, n);
        for i in 0..n {
            m.set(i, i, Rational::ONE);
        }
        m
    }

    /// Row-reduces an augmented `n × (n + extra)` copy to reduced row echelon
    /// form, returning the number of pivots found and, when full-rank, the
    /// accumulated determinant of the left block. Works in place on `data`.
    fn eliminate(
        data: &mut [crate::rational::Rational],
        n: usize,
        width: usize,
    ) -> (usize, crate::rational::Rational) {
        use crate::rational::Rational;
        let mut det = Rational::ONE;
        let mut pivots = 0;
        for col in 0..n {
            let piv = (pivots..n).find(|&r| !data[r * width + col].is_zero());
            let piv = match piv {
                Some(p) => p,
                None => {
                    det = Rational::ZERO;
                    continue;
                }
            };
            if piv != pivots {
                for c in 0..width {
                    data.swap(pivots * width + c, piv * width + c);
                }
                det = Rational::neg(&det);
            }
            let pivot = data[pivots * width + col].clone();
            det = Rational::mul(&det, &pivot);
            // Normalize the pivot row.
            for c in 0..width {
                data[pivots * width + c] = Rational::div(&data[pivots * width + c], &pivot);
            }
            // Eliminate the column from all other rows.
            for r in 0..n {
                if r == pivots {
                    continue;
                }
                let factor = data[r * width + col].clone();
                if factor.is_zero() {
                    continue;
                }
                for c in 0..width {
                    let prod = Rational::mul(&factor, &data[pivots * width + c]);
                    data[r * width + c] = Rational::sub(&data[r * width + c], &prod);
                }
            }
            pivots += 1;
        }
        (pivots, det)
    }

    /// Returns the exact determinant. Panics if the matrix is not square.
    pub fn determinant(&self) -> crate::rational::Rational {
        use crate::int::Int;
        use crate::rational::Rational;
        assert!(self.is_square(), "determinant: matrix must be square");
        let n = self.rows;
        if n == 0 {
            return Rational::ONE;
        }
        // Clear denominators row by row into an integer matrix (scaling row i by
        // sᵢ = lcm of its denominators multiplies the determinant by sᵢ), then use
        // the fraction-free integer Bareiss determinant — whose intermediate
        // entries stay bounded (Hadamard) — instead of rational Gaussian
        // elimination, which suffers numerator/denominator blow-up.
        let mut int_data = alloc::vec::Vec::with_capacity(n * n);
        let mut scale = Int::ONE;
        for i in 0..n {
            let mut s = Int::ONE;
            for j in 0..n {
                s = s.lcm(self.get(i, j).denominator());
            }
            for j in 0..n {
                let e = self.get(i, j);
                let factor = s.div_exact(e.denominator()); // exact: denominator | s
                int_data.push(e.numerator().mul(&factor));
            }
            scale = scale.mul(&s);
        }
        let int_det = Matrix::<Int>::new(n, n, int_data).determinant();
        Rational::new(int_det, scale)
    }

    /// Returns the rank (number of linearly independent rows).
    pub fn rank(&self) -> usize {
        use crate::rational::Rational;
        if self.rows == 0 || self.cols == 0 {
            return 0;
        }
        // Eliminate over min(rows, cols) pivot columns using a padded copy.
        let n = self.rows;
        let width = self.cols;
        let mut data = self.data.clone();
        let mut pivots = 0;
        for col in 0..self.cols {
            if pivots == n {
                break;
            }
            let piv = (pivots..n).find(|&r| !data[r * width + col].is_zero());
            let piv = match piv {
                Some(p) => p,
                None => continue,
            };
            if piv != pivots {
                for c in 0..width {
                    data.swap(pivots * width + c, piv * width + c);
                }
            }
            let pivot = data[pivots * width + col].clone();
            for r in pivots + 1..n {
                let factor = Rational::div(&data[r * width + col], &pivot);
                for c in col..width {
                    let prod = Rational::mul(&factor, &data[pivots * width + c]);
                    data[r * width + c] = Rational::sub(&data[r * width + c], &prod);
                }
            }
            pivots += 1;
        }
        pivots
    }

    /// Returns the inverse, or `None` if the matrix is singular. Panics if not
    /// square.
    pub fn inverse(&self) -> Option<Matrix<crate::rational::Rational>> {
        use crate::rational::Rational;
        assert!(self.is_square(), "inverse: matrix must be square");
        let n = self.rows;
        let width = 2 * n;
        // Augmented [A | I].
        let mut data = alloc::vec![Rational::ZERO; n * width];
        for i in 0..n {
            for j in 0..n {
                data[i * width + j] = self.data[i * n + j].clone();
            }
            data[i * width + n + i] = Rational::ONE;
        }
        // Fraction-free path (fast); fall back to rational Gauss-Jordan only if a
        // zero pivot forces a row swap.
        if let Some(sol) = fraction_free_solve(&data, n, n) {
            let mut inv = Matrix::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    inv.set(i, j, sol[i * n + j].clone());
                }
            }
            return Some(inv);
        }
        let (pivots, _) = Self::eliminate(&mut data, n, width);
        if pivots < n {
            return None; // singular
        }
        let mut inv = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                inv.set(i, j, data[i * width + n + j].clone());
            }
        }
        Some(inv)
    }

    /// Solves `self · x = b`, returning `x`, or `None` if there is no unique
    /// solution. Panics if `self` is not square or `b` has the wrong length.
    pub fn solve(&self, b: &[crate::rational::Rational]) -> Option<Vec<crate::rational::Rational>> {
        use crate::rational::Rational;
        assert!(self.is_square(), "solve: matrix must be square");
        let n = self.rows;
        assert_eq!(b.len(), n, "solve: right-hand side length mismatch");
        let width = n + 1;
        let mut data = alloc::vec![Rational::ZERO; n * width];
        for i in 0..n {
            for j in 0..n {
                data[i * width + j] = self.data[i * n + j].clone();
            }
            data[i * width + n] = b[i].clone();
        }
        if let Some(sol) = fraction_free_solve(&data, n, 1) {
            return Some(sol); // already n×1, row-major
        }
        let (pivots, _) = Self::eliminate(&mut data, n, width);
        if pivots < n {
            return None;
        }
        Some((0..n).map(|i| data[i * width + n].clone()).collect())
    }
}

/// Solves the augmented rational system `[A | R]` (A in columns `0..n`, the
/// `extra` right-hand sides in columns `n..n+extra`) fraction-free: clear each
/// row's denominators to integers, run Bareiss forward elimination (exact
/// division by the previous pivot keeps intermediate entries Hadamard-bounded
/// instead of blowing up like rational Gaussian elimination), then
/// back-substitute in the rationals. Returns the `n × extra` solution row-major,
/// or `None` if a zero pivot is hit (singular, or would need a row swap) — the
/// caller falls back to the exact rational path in that case.
#[cfg(feature = "rational")]
fn fraction_free_solve(
    aug: &[crate::rational::Rational],
    n: usize,
    extra: usize,
) -> Option<alloc::vec::Vec<crate::rational::Rational>> {
    use crate::int::Int;
    use crate::rational::Rational;
    let width = n + extra;

    // Clear each row's denominators (including its right-hand sides) → integers.
    let mut m: alloc::vec::Vec<Int> = alloc::vec::Vec::with_capacity(n * width);
    for i in 0..n {
        let mut s = Int::ONE;
        for j in 0..width {
            s = s.lcm(aug[i * width + j].denominator());
        }
        for j in 0..width {
            let e = &aug[i * width + j];
            m.push(e.numerator().mul(&s.div_exact(e.denominator())));
        }
    }

    // Bareiss forward elimination (no row swaps — bail to the caller on a zero
    // pivot so correctness never depends on fraction-free pivoting subtleties).
    let mut prev = Int::ONE;
    for k in 0..n {
        if m[k * width + k].is_zero() {
            return None;
        }
        let mkk = m[k * width + k].clone();
        for i in k + 1..n {
            let mik = m[i * width + k].clone();
            for j in k + 1..width {
                let num = Int::sub(
                    &Int::mul(&mkk, &m[i * width + j]),
                    &Int::mul(&mik, &m[k * width + j]),
                );
                m[i * width + j] = num.div_exact(&prev); // exact by the Bareiss identity
            }
            m[i * width + k] = Int::ZERO;
        }
        prev = mkk;
    }

    // Rational back-substitution of each right-hand column.
    let mut x = alloc::vec![Rational::ZERO; n * extra];
    for c in 0..extra {
        for i in (0..n).rev() {
            let mut acc = Rational::from_integer(m[i * width + n + c].clone());
            for j in i + 1..n {
                let term = Rational::mul(
                    &Rational::from_integer(m[i * width + j].clone()),
                    &x[j * extra + c],
                );
                acc = Rational::sub(&acc, &term);
            }
            x[i * extra + c] =
                Rational::div(&acc, &Rational::from_integer(m[i * width + i].clone()));
        }
    }
    Some(x)
}

/// Generic exact linear algebra over an arbitrary [`Field`], by **Gaussian
/// elimination with partial pivoting**.
///
/// This trait unlocks determinant / inverse / linear-solve / rank for every
/// [`Matrix<T>`] whose component type `T` is a [`Field`] — in particular the
/// finite fields `GF(p)` ([`ModInt`](crate::mod_int::ModInt) with a prime
/// modulus), `GF(pᵏ)` ([`GfElement`](crate::galois::GfElement)), and
/// [`Float`](crate::float::Float) — none of which have a bespoke inherent
/// algorithm.
///
/// # Coherence with the concrete `Int` / `Rational` algorithms
///
/// [`Matrix<Rational>`](Matrix) already exposes *inherent* `determinant`,
/// `inverse`, `solve`, and `rank` (the optimized fraction-free / Bareiss exact
/// path). Because `Rational: Field`, this trait is also implemented for it — but
/// Rust resolves plain method calls to an **inherent method in preference to a
/// trait method**, so `m.determinant()` on a `Matrix<Rational>` keeps using the
/// fast Bareiss path. To invoke *this* generic Gaussian implementation on such a
/// matrix, call it through the trait explicitly, e.g.
/// `FieldMatrix::determinant(&m)`. ([`Matrix<Int>`](Matrix) is unaffected:
/// `Int` is not a `Field`, so the trait is not implemented for it.)
///
/// Bring the trait into scope with `use puremp::FieldMatrix;` to call these
/// methods on `Matrix<ModInt>`, `Matrix<GfElement>`, `Matrix<Float>`, etc.
///
/// # Caveats
///
/// - There is no fraction-free variant here; every step divides by the pivot
///   using the field's [`Div`](core::ops::Div) / [`Field::inv`], so this path is
///   meant for fields where that is cheap and exact (finite fields), not for
///   `Rational` (use its inherent Bareiss path).
/// - Over [`ModInt`](crate::mod_int::ModInt) the modulus must be **prime** for
///   the ring to actually be a field; a nonzero non-invertible pivot would
///   otherwise make division panic.
pub trait FieldMatrix<T: Field> {
    /// The determinant, computed as `(∏ pivots) · (−1)^{#row swaps}` after
    /// forward elimination to upper-triangular form. A wholly-zero pivot column
    /// means the matrix is singular, so the determinant is the field's zero.
    /// Panics if the matrix is not square (or is `0 × 0`, which has no sample
    /// element from which to take the ring's one).
    fn determinant(&self) -> T;

    /// The inverse via Gauss–Jordan elimination on the augmented `[A | I]`, or
    /// `None` if `A` is singular. Panics if the matrix is not square.
    fn inverse(&self) -> Option<Matrix<T>>;

    /// Solves `self · x = b`, returning `x`, or `None` when there is no unique
    /// solution (a zero pivot column ⇒ singular). Panics if `self` is not square
    /// or `b` has the wrong length.
    fn solve(&self, b: &[T]) -> Option<Vec<T>>;

    /// The rank: the number of nonzero pivots produced by row-reducing to row
    /// echelon form (works for any shape).
    fn rank(&self) -> usize;
}

impl<T: Field> FieldMatrix<T> for Matrix<T> {
    fn determinant(&self) -> T {
        assert!(self.is_square(), "determinant: matrix must be square");
        let n = self.rows;
        let sample = self
            .data
            .first()
            .expect("determinant: empty matrix has no sample element for the ring's one");
        let zero = sample.zero();
        let mut det = sample.one();
        let mut a = self.data.clone();
        let idx = |i: usize, j: usize| i * n + j;
        let mut negated = false;
        for col in 0..n {
            // Choose any nonzero pivot at or below the diagonal in this column.
            let piv = match (col..n).find(|&r| !a[idx(r, col)].is_zero()) {
                Some(p) => p,
                None => return zero, // whole sub-column zero ⇒ singular
            };
            if piv != col {
                for c in 0..n {
                    a.swap(idx(col, c), idx(piv, c));
                }
                negated = !negated;
            }
            let pivot = a[idx(col, col)].clone();
            det = det * pivot.clone();
            for r in col + 1..n {
                let factor = a[idx(r, col)].clone() / pivot.clone();
                if factor.is_zero() {
                    continue;
                }
                for c in col..n {
                    let prod = factor.clone() * a[idx(col, c)].clone();
                    a[idx(r, c)] = a[idx(r, c)].clone() - prod;
                }
            }
        }
        if negated { -det } else { det }
    }

    fn inverse(&self) -> Option<Matrix<T>> {
        assert!(self.is_square(), "inverse: matrix must be square");
        let n = self.rows;
        if n == 0 {
            return Some(self.clone()); // the empty matrix is its own inverse
        }
        let sample = &self.data[0];
        let zero = sample.zero();
        let one = sample.one();
        let width = 2 * n;
        // Augmented [A | I].
        let mut data = alloc::vec![zero.clone(); n * width];
        for i in 0..n {
            for j in 0..n {
                data[i * width + j] = self.data[i * n + j].clone();
            }
            data[i * width + n + i] = one.clone();
        }
        for col in 0..n {
            let piv = (col..n).find(|&r| !data[r * width + col].is_zero())?; // singular
            if piv != col {
                for c in 0..width {
                    data.swap(col * width + c, piv * width + c);
                }
            }
            let pivot = data[col * width + col].clone();
            // Normalize the pivot row.
            for c in 0..width {
                data[col * width + c] = data[col * width + c].clone() / pivot.clone();
            }
            // Eliminate the column from every other row.
            for r in 0..n {
                if r == col {
                    continue;
                }
                let factor = data[r * width + col].clone();
                if factor.is_zero() {
                    continue;
                }
                for c in 0..width {
                    let prod = factor.clone() * data[col * width + c].clone();
                    data[r * width + c] = data[r * width + c].clone() - prod;
                }
            }
        }
        let mut inv = Matrix::filled(zero, n, n);
        for i in 0..n {
            for j in 0..n {
                inv.set(i, j, data[i * width + n + j].clone());
            }
        }
        Some(inv)
    }

    fn solve(&self, b: &[T]) -> Option<Vec<T>> {
        assert!(self.is_square(), "solve: matrix must be square");
        let n = self.rows;
        assert_eq!(b.len(), n, "solve: right-hand side length mismatch");
        if n == 0 {
            return Some(Vec::new());
        }
        let zero = self.data[0].zero();
        let width = n + 1;
        let mut data = alloc::vec![zero.clone(); n * width];
        for i in 0..n {
            for j in 0..n {
                data[i * width + j] = self.data[i * n + j].clone();
            }
            data[i * width + n] = b[i].clone();
        }
        // Forward-eliminate to upper triangular.
        for col in 0..n {
            let piv = (col..n).find(|&r| !data[r * width + col].is_zero())?; // singular
            if piv != col {
                for c in 0..width {
                    data.swap(col * width + c, piv * width + c);
                }
            }
            let pivot = data[col * width + col].clone();
            for r in col + 1..n {
                let factor = data[r * width + col].clone() / pivot.clone();
                if factor.is_zero() {
                    continue;
                }
                for c in col..width {
                    let prod = factor.clone() * data[col * width + c].clone();
                    data[r * width + c] = data[r * width + c].clone() - prod;
                }
            }
        }
        // Back-substitution.
        let mut x = alloc::vec![zero; n];
        for i in (0..n).rev() {
            let mut acc = data[i * width + n].clone();
            for j in i + 1..n {
                acc = acc - data[i * width + j].clone() * x[j].clone();
            }
            x[i] = acc / data[i * width + i].clone();
        }
        Some(x)
    }

    fn rank(&self) -> usize {
        if self.rows == 0 || self.cols == 0 {
            return 0;
        }
        let n = self.rows;
        let width = self.cols;
        let mut data = self.data.clone();
        let mut pivots = 0;
        for col in 0..self.cols {
            if pivots == n {
                break;
            }
            let piv = match (pivots..n).find(|&r| !data[r * width + col].is_zero()) {
                Some(p) => p,
                None => continue,
            };
            if piv != pivots {
                for c in 0..width {
                    data.swap(pivots * width + c, piv * width + c);
                }
            }
            let pivot = data[pivots * width + col].clone();
            for r in pivots + 1..n {
                let factor = data[r * width + col].clone() / pivot.clone();
                if factor.is_zero() {
                    continue;
                }
                for c in col..width {
                    let prod = factor.clone() * data[pivots * width + c].clone();
                    data[r * width + c] = data[r * width + c].clone() - prod;
                }
            }
            pivots += 1;
        }
        pivots
    }
}

macro_rules! matrix_binop {
    ($tr:ident, $m:ident, $atr:ident, $am:ident) => {
        impl<T: Ring> core::ops::$tr for Matrix<T> {
            type Output = Matrix<T>;
            #[inline]
            fn $m(self, rhs: Matrix<T>) -> Matrix<T> {
                Matrix::$m(&self, &rhs)
            }
        }
        impl<T: Ring> core::ops::$tr<&Matrix<T>> for &Matrix<T> {
            type Output = Matrix<T>;
            #[inline]
            fn $m(self, rhs: &Matrix<T>) -> Matrix<T> {
                Matrix::$m(self, rhs)
            }
        }
    };
}

matrix_binop!(Add, add, AddAssign, add_assign);
matrix_binop!(Sub, sub, SubAssign, sub_assign);
matrix_binop!(Mul, mul, MulAssign, mul_assign);

/// Division-free linear algebra over any commutative [`Ring`] — the
/// characteristic polynomial and determinant by the **Samuelson–Berkowitz**
/// algorithm, which uses only ring `+`/`−`/`×` (no division at all). This works
/// over rings that are not fields or integral domains — `ModInt` with a composite
/// modulus, `Matrix<Poly<Int>>`, and the like — where Gaussian elimination and
/// fraction-free Bareiss cannot be applied.
///
/// For `Int`/`Rational` the inherent [`Matrix::determinant`] (Bareiss) is faster,
/// and for a genuine field [`FieldMatrix::determinant`] (Gaussian) is faster;
/// these are the universal fallback. Cost is `O(n⁴)` ring multiplications.
pub trait RingMatrix<T: Ring> {
    /// The characteristic polynomial `det(x·I − self)` as coefficients from the
    /// constant term up: `charpoly()[i]` is the coefficient of `xⁱ`, so
    /// `charpoly()[0]` is `(−1)ⁿ·det` and the leading coefficient is one.
    ///
    /// # Panics
    /// If the matrix is not square, or is `0×0` (no element to source the ring).
    fn charpoly(&self) -> alloc::vec::Vec<T>;

    /// The determinant of `self`, computed division-free.
    ///
    /// # Panics
    /// If the matrix is not square, or is `0×0`.
    fn det(&self) -> T;
}

impl<T: Ring> RingMatrix<T> for Matrix<T> {
    #[allow(clippy::needless_range_loop)] // `j` is the power in S·Mʲ⁻²·R, not just an index
    fn charpoly(&self) -> alloc::vec::Vec<T> {
        assert!(
            self.is_square(),
            "RingMatrix::charpoly: matrix must be square"
        );
        let n = self.rows();
        assert!(
            n > 0,
            "RingMatrix::charpoly: 0×0 matrix has no ring context"
        );
        let one = self.get(0, 0).one();
        let zero = self.get(0, 0).zero();
        // `v` = characteristic polynomial of the leading r×r block, high-degree
        // first (`v[0]` is the coefficient of xʳ = one), grown by a Toeplitz
        // matrix–vector product each step (Samuelson–Berkowitz).
        let mut v = alloc::vec![one.clone()];
        for r in 1..=n {
            let a = self.get(r - 1, r - 1).clone();
            // Toeplitz first column `t` (length r+1): t0 = 1, t1 = −a,
            // t_j = −(S·Mʲ⁻²·R) for j = 2..=r, where M is the leading (r−1)² block,
            // R = column r−1 (rows 0..r−1), S = row r−1 (cols 0..r−1).
            let mut t = alloc::vec![zero.clone(); r + 1];
            t[0] = one.clone();
            t[1] = -a.clone();
            if r >= 2 {
                let mut w: alloc::vec::Vec<T> =
                    (0..r - 1).map(|i| self.get(i, r - 1).clone()).collect();
                for j in 2..=r {
                    let mut s = zero.clone();
                    for (c, wc) in w.iter().enumerate() {
                        s = s + self.get(r - 1, c).clone() * wc.clone();
                    }
                    t[j] = -s;
                    if j < r {
                        let mut wn = alloc::vec![zero.clone(); r - 1];
                        for (i, wni) in wn.iter_mut().enumerate() {
                            let mut acc = zero.clone();
                            for (k, wk) in w.iter().enumerate() {
                                acc = acc + self.get(i, k).clone() * wk.clone();
                            }
                            *wni = acc;
                        }
                        w = wn;
                    }
                }
            }
            // v_new = T · v, with T[(i,k)] = t[i−k] (lower-triangular Toeplitz).
            let mut vn = alloc::vec![zero.clone(); r + 1];
            for (i, vni) in vn.iter_mut().enumerate() {
                let mut acc = zero.clone();
                for (k, vk) in v.iter().enumerate() {
                    if i >= k {
                        acc = acc + t[i - k].clone() * vk.clone();
                    }
                }
                *vni = acc;
            }
            v = vn;
        }
        v.reverse(); // high-to-low → low-to-high (index = power of x)
        v
    }

    fn det(&self) -> T {
        let c = self.charpoly();
        // det = (−1)ⁿ · constant term.
        if self.rows().is_multiple_of(2) {
            c[0].clone()
        } else {
            -c[0].clone()
        }
    }
}

// ---- exact eigenvalues of a rational matrix ----

/// Exact eigenvalues of a rational matrix, as real algebraic numbers.
///
/// These methods compose the two exact building blocks the crate already
/// provides: the division-free [`RingMatrix::charpoly`] (Samuelson–Berkowitz)
/// gives the monic characteristic polynomial `det(x·I − A)` over ℚ, and
/// [`Algebraic::real_roots_of`](crate::algebraic::Algebraic::real_roots_of)
/// isolates its real roots exactly (Sturm sequences + bisection). Every returned
/// eigenvalue is therefore an exact [`Algebraic`](crate::algebraic::Algebraic) —
/// compared and combined by its true real value, never a float approximation.
///
/// # Real eigenvalues only
///
/// [`Algebraic`](crate::algebraic::Algebraic) models *real* algebraic numbers, so
/// only the **real** eigenvalues are returned; non-real (complex-conjugate)
/// eigenvalues are silently omitted. A matrix whose spectrum is entirely complex
/// (e.g. the rotation `[[0,−1],[1,0]]`, eigenvalues `±i`) yields an empty list
/// without error.
///
/// # Cost
///
/// The characteristic polynomial costs `O(n⁴)` rational multiplications
/// (Samuelson–Berkowitz), and real-root isolation runs Sturm sequences and
/// polynomial GCDs by the subresultant PRS on the degree-`n` char poly. Both are
/// exact but grow quickly with `n`; this is intended for modest matrix sizes.
#[cfg(feature = "algebraic")]
impl Matrix<crate::rational::Rational> {
    /// Returns the characteristic polynomial `det(x·I − A)` over ℚ, monic and in
    /// low-to-high coefficient order (`coeffs()[i]` is the coefficient of `xⁱ`).
    ///
    /// Computed division-free via [`RingMatrix::charpoly`]. Panics if the matrix
    /// is not square (or is `0×0`, which has no ring context).
    pub fn characteristic_polynomial(&self) -> crate::poly::Poly<crate::rational::Rational> {
        crate::poly::Poly::new(RingMatrix::charpoly(self))
    }

    /// Returns the **distinct real eigenvalues** as exact algebraic numbers, in
    /// increasing order.
    ///
    /// Non-real (complex) eigenvalues are not returned — see the
    /// [type-level note](Matrix#real-eigenvalues-only). A repeated eigenvalue
    /// appears once; use
    /// [`real_eigenvalues_with_multiplicity`](Self::real_eigenvalues_with_multiplicity)
    /// for algebraic multiplicities.
    ///
    /// Panics if the matrix is not square (or is `0×0`).
    pub fn real_eigenvalues(&self) -> Vec<crate::algebraic::Algebraic> {
        crate::algebraic::Algebraic::real_roots_of(&self.characteristic_polynomial())
    }

    /// Returns the distinct real eigenvalues paired with their **algebraic
    /// multiplicity** (their multiplicity as roots of the characteristic
    /// polynomial), in increasing order of eigenvalue.
    ///
    /// The multiplicities come from a squarefree decomposition of the char poly
    /// (Yun's algorithm — repeated GCDs with the derivative): the roots of the
    /// multiplicity-`k` squarefree factor are exactly the eigenvalues of algebraic
    /// multiplicity `k`. Non-real eigenvalues are omitted, so the returned
    /// multiplicities need not sum to `n`.
    ///
    /// Panics if the matrix is not square (or is `0×0`).
    pub fn real_eigenvalues_with_multiplicity(&self) -> Vec<(crate::algebraic::Algebraic, usize)> {
        use crate::algebraic::Algebraic;
        let cp = self.characteristic_polynomial();
        let mut out = Vec::new();
        for (i, factor) in squarefree_decomposition(&cp).into_iter().enumerate() {
            let mult = i + 1; // factor i (0-based) holds the roots of multiplicity i+1
            for root in Algebraic::real_roots_of(&factor) {
                out.push((root, mult));
            }
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }
}

/// Squarefree decomposition of a rational polynomial by **Yun's algorithm**.
///
/// Returns `[g₁, g₂, …]` where `gₖ` (monic) is the product of the distinct
/// irreducible factors of `p` that occur with multiplicity exactly `k`; entry
/// `k−1` may be a constant (`1`) when no factor has that multiplicity. Uses only
/// derivative, subresultant GCD, and exact division — no full factorization.
#[cfg(feature = "algebraic")]
fn squarefree_decomposition(
    p: &crate::poly::Poly<crate::rational::Rational>,
) -> Vec<crate::poly::Poly<crate::rational::Rational>> {
    let p = p.monic();
    if p.degree().unwrap_or(0) < 1 {
        return Vec::new();
    }
    let d = p.derivative();
    let a0 = p.subresultant_gcd(&d);
    let mut b = p.div_rem(&a0).0; // b₁ = p / gcd(p, p′)
    let mut c = d.div_rem(&a0).0; // c₁ = p′ / gcd(p, p′)
    let mut result = Vec::new();
    loop {
        let dd = c.sub(&b.derivative()); // dₖ = cₖ − bₖ′
        let g = b.subresultant_gcd(&dd); // gₖ = gcd(bₖ, dₖ): the multiplicity-k factor
        result.push(g.monic());
        b = b.div_rem(&g).0; // bₖ₊₁ = bₖ / gₖ
        c = dd.div_rem(&g).0; // cₖ₊₁ = dₖ / gₖ
        if b.degree().unwrap_or(0) < 1 {
            break;
        }
    }
    result
}

#[cfg(all(test, feature = "rational"))]
mod strassen_tests {
    use super::*;
    use crate::int::Int;
    use crate::rational::Rational;

    /// A tiny deterministic LCG — enough to build reproducible random matrices.
    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Lcg {
            Lcg(seed)
        }
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 11
        }
        /// A signed integer of roughly `bits` bits.
        fn int(&mut self, bits: u32) -> Int {
            let mut v = Int::ZERO;
            let chunk = Int::from_i64(1i64 << 20);
            let mut b = 0u32;
            while b < bits {
                v = &(&v * &chunk) + &Int::from_i64((self.next() & 0xF_FFFF) as i64);
                b += 20;
            }
            if self.next() & 1 == 0 { v.neg() } else { v }
        }
        fn imatrix(&mut self, rows: usize, cols: usize, bits: u32) -> Matrix<Int> {
            let data = (0..rows * cols).map(|_| self.int(bits)).collect();
            Matrix::new(rows, cols, data)
        }
        fn rmatrix(&mut self, rows: usize, cols: usize, bits: u32) -> Matrix<Rational> {
            let data = (0..rows * cols)
                .map(|_| {
                    let d = self.int(bits.max(1));
                    let den = if d.is_zero() { Int::ONE } else { d };
                    Rational::new(self.int(bits), den)
                })
                .collect();
            Matrix::new(rows, cols, data)
        }
    }

    /// Shapes covering: below/at/above the threshold, odd dimensions,
    /// rectangular, degenerate, and two recursion levels (dim > 48). Kept ≤ 65
    /// so the differential runs quickly in unoptimized `cargo test`.
    const SHAPES: &[(usize, usize, usize)] = &[
        (1, 1, 1),
        (2, 2, 2),
        (3, 5, 4),
        (24, 24, 24),
        (25, 25, 25),
        (26, 26, 26),
        (25, 26, 27),
        (33, 17, 49),
        (48, 48, 48),
        (49, 49, 49),
        (50, 37, 63),
        (65, 65, 65),
        (1, 60, 1),
        (60, 1, 60),
    ];

    #[test]
    fn strassen_matches_naive_int_small_entries() {
        let mut r = Lcg::new(0xC0FFEE);
        for &(m, k, n) in SHAPES {
            let a = r.imatrix(m, k, 30);
            let b = r.imatrix(k, n, 30);
            assert_eq!(
                a.strassen_mul(&b),
                a.naive_mul(&b),
                "int mismatch at {m}x{k} * {k}x{n}"
            );
        }
    }

    #[test]
    fn strassen_matches_naive_int_large_entries() {
        let mut r = Lcg::new(0x1234_5678);
        // Big entries (~300 bits) across odd / rectangular / two-level shapes —
        // exercises carries and signs without the debug-mode cost of huge dims.
        for &(m, k, n) in &[(26usize, 26, 26), (33usize, 25, 41)] {
            let a = r.imatrix(m, k, 300);
            let b = r.imatrix(k, n, 300);
            assert_eq!(a.strassen_mul(&b), a.naive_mul(&b));
        }
    }

    #[test]
    fn strassen_matches_naive_rational() {
        let mut r = Lcg::new(0xABCD_EF01);
        // Rational multiplies (gcd-heavy) are dear in debug, so use a smaller
        // subset of shapes that still spans odd / rectangular / two levels.
        for &(m, k, n) in &[(2usize, 2, 2), (25usize, 25, 25), (33usize, 17, 49)] {
            let a = r.rmatrix(m, k, 40);
            let b = r.rmatrix(k, n, 40);
            assert_eq!(
                a.strassen_mul(&b),
                a.naive_mul(&b),
                "rational mismatch at {m}x{k} * {k}x{n}"
            );
        }
    }

    #[test]
    fn public_mul_dispatches_and_matches_naive_large() {
        // Entries above the bit cutoff so the public `mul` really takes the
        // Strassen path; it must still equal the naive product bit-for-bit.
        let mut r = Lcg::new(0x55AA_55AA);
        let a = r.imatrix(26, 26, 1100);
        let b = r.imatrix(26, 26, 1100);
        assert!(a.get(0, 0).multiply_cost_hint() >= STRASSEN_MIN_ENTRY_BITS);
        assert_eq!(a.mul(&b), a.naive_mul(&b));
        // A · I = A on the Strassen path.
        let id = Matrix::<Int>::identity(26);
        assert_eq!(a.mul(&id), a);
    }
}

#[cfg(all(test, feature = "int"))]
mod hnf_snf_tests {
    use super::*;
    use crate::int::Int;

    /// Tiny deterministic LCG for reproducible random test matrices.
    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Lcg {
            Lcg(seed)
        }
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 11
        }
        /// A signed integer in `[-range, range]`.
        fn small(&mut self, range: i64) -> Int {
            let span = (2 * range + 1) as u64;
            Int::from_i64((self.next() % span) as i64 - range)
        }
        fn imatrix(&mut self, rows: usize, cols: usize, range: i64) -> Matrix<Int> {
            let data = (0..rows * cols).map(|_| self.small(range)).collect();
            Matrix::new(rows, cols, data)
        }
        /// A signed integer of roughly `nbits` bits.
        fn bits(&mut self, nbits: u32) -> Int {
            let mut v = Int::ZERO;
            let mut got = 0u32;
            while got < nbits {
                v = v.mul_2k(32).add(&Int::from_u64(self.next() & 0xFFFF_FFFF));
                got += 32;
            }
            if self.next() & 1 == 0 { v.neg() } else { v }
        }
        /// A full-(row-)rank `rows × cols` matrix (`rows ≤ cols`) with ~`nbits`
        /// entries, by rejection sampling on the rank.
        fn full_rank(&mut self, rows: usize, cols: usize, nbits: u32) -> Matrix<Int> {
            loop {
                let data = (0..rows * cols).map(|_| self.bits(nbits)).collect();
                let a = Matrix::new(rows, cols, data);
                if a.rank() == rows {
                    return a;
                }
            }
        }
        /// A random unimodular `n × n` matrix built from `steps` elementary row
        /// operations on the identity (transvections, swaps, negations).
        fn unimodular(&mut self, n: usize, steps: usize) -> Matrix<Int> {
            let mut m = Matrix::<Int>::identity(n);
            for _ in 0..steps {
                if n < 2 {
                    break;
                }
                let i = (self.next() as usize) % n;
                let mut j = (self.next() as usize) % n;
                if i == j {
                    j = (j + 1) % n;
                }
                match self.next() % 3 {
                    0 => {
                        // row_i += q · row_j
                        let q = self.small(3);
                        for c in 0..n {
                            let add = q.mul(m.get(j, c));
                            let v = m.get(i, c).add(&add);
                            m.set(i, c, v);
                        }
                    }
                    1 => {
                        for c in 0..n {
                            let a = m.get(i, c).clone();
                            let b = m.get(j, c).clone();
                            m.set(i, c, b);
                            m.set(j, c, a);
                        }
                    }
                    _ => {
                        for c in 0..n {
                            let v = m.get(i, c).neg();
                            m.set(i, c, v);
                        }
                    }
                }
            }
            m
        }
    }

    /// Verifies the row-HNF shape conditions and returns the pivot list.
    fn check_hnf_shape(h: &Matrix<Int>) -> Vec<(usize, Int)> {
        let m = h.rows();
        let n = h.cols();
        let mut pivots = Vec::new();
        let mut prev_col: Option<usize> = None;
        let mut seen_zero_row = false;
        for i in 0..m {
            let lead = (0..n).find(|&j| !h.get(i, j).is_zero());
            match lead {
                None => seen_zero_row = true,
                Some(j) => {
                    assert!(!seen_zero_row, "nonzero row {i} after a zero row");
                    if let Some(pc) = prev_col {
                        assert!(j > pc, "pivot columns not strictly increasing at row {i}");
                    }
                    prev_col = Some(j);
                    let piv = h.get(i, j).clone();
                    assert!(piv.is_positive(), "pivot at ({i},{j}) not positive");
                    for r in 0..i {
                        let e = h.get(r, j);
                        assert!(
                            !e.is_negative() && e < &piv,
                            "entry above pivot ({r},{j}) not reduced into [0,piv)"
                        );
                    }
                    pivots.push((j, piv));
                }
            }
        }
        pivots
    }

    fn is_unimodular(u: &Matrix<Int>) -> bool {
        u.is_square() && u.determinant().abs().is_one()
    }

    /// Verifies SNF diagonal shape and returns the invariant factors.
    fn check_snf_shape(d: &Matrix<Int>) -> Vec<Int> {
        let m = d.rows();
        let n = d.cols();
        for i in 0..m {
            for j in 0..n {
                if i != j {
                    assert!(d.get(i, j).is_zero(), "off-diagonal ({i},{j}) nonzero");
                }
            }
        }
        let mut factors = Vec::new();
        let mut seen_zero = false;
        for i in 0..m.min(n) {
            let e = d.get(i, i).clone();
            if e.is_zero() {
                seen_zero = true;
            } else {
                assert!(!seen_zero, "nonzero invariant factor after a zero one");
                assert!(e.is_positive(), "invariant factor not positive");
                factors.push(e);
            }
        }
        for w in factors.windows(2) {
            assert!(w[0].divides(&w[1]), "divisibility d_i | d_{{i+1}} fails");
        }
        factors
    }

    fn zero_vec(len: usize) -> Vec<Int> {
        alloc::vec![Int::ZERO; len]
    }

    /// Full HNF contract on one matrix: `H = U·A`, `U` unimodular, shape valid,
    /// canonical under row-equivalence, and diagonal-product = |det| when square.
    fn assert_hnf_contract(a: &Matrix<Int>, rng: &mut Lcg) {
        let (h, u) = a.hermite_normal_form_with_transform();
        assert_eq!(
            h,
            a.hermite_normal_form(),
            "with/without transform disagree"
        );
        assert!(is_unimodular(&u), "U not unimodular");
        assert_eq!(u.mul(a), h, "H = U·A violated");
        let pivots = check_hnf_shape(&h);
        assert_eq!(pivots.len(), a.rank(), "rank mismatch vs pivot count");

        // Canonical form: a row-equivalent matrix has the same HNF.
        let w = rng.unimodular(a.rows(), 12);
        let a2 = w.mul(a);
        assert_eq!(h, a2.hermite_normal_form(), "HNF not canonical");

        if a.is_square() {
            let det = a.determinant();
            if !det.is_zero() {
                let mut prod = Int::ONE;
                for (_, p) in &pivots {
                    prod = prod.mul(p);
                }
                assert_eq!(prod, det.abs(), "∏ pivots ≠ |det|");
            }
        }
    }

    /// Full SNF contract on one matrix: `D = U·A·V`, transforms unimodular, shape
    /// valid, `d₀ = gcd(entries)`, and `∏ dᵢ = |det|` when square nonsingular.
    fn assert_snf_contract(a: &Matrix<Int>) {
        let (u, d, v) = a.smith_normal_form_with_transforms();
        assert_eq!(d, a.smith_normal_form(), "with/without transforms disagree");
        assert!(is_unimodular(&u), "U not unimodular");
        assert!(is_unimodular(&v), "V not unimodular");
        assert_eq!(u.mul(a).mul(&v), d, "D = U·A·V violated");
        let factors = check_snf_shape(&d);
        assert_eq!(factors, a.invariant_factors(), "invariant_factors mismatch");
        assert_eq!(
            factors.len(),
            a.rank(),
            "rank mismatch vs #invariant factors"
        );

        // d₀ is the gcd of all entries.
        if !factors.is_empty() {
            let mut g = Int::ZERO;
            for e in a.as_slice() {
                g = g.gcd(e);
            }
            assert_eq!(factors[0], g, "d₀ ≠ gcd of entries");
        }

        if a.is_square() {
            let det = a.determinant();
            if !det.is_zero() {
                let mut prod = Int::ONE;
                for f in &factors {
                    prod = prod.mul(f);
                }
                assert_eq!(prod, det.abs(), "∏ dᵢ ≠ |det|");
            }
        }
    }

    /// Verifies the kernel basis: each vector `k` satisfies `A·k = 0`, and the
    /// count is `cols − rank`.
    fn assert_kernel(a: &Matrix<Int>) {
        let ker = a.kernel();
        let n = a.cols();
        assert_eq!(ker.len(), n - a.rank(), "kernel dimension wrong");
        for k in &ker {
            assert_eq!(k.len(), n);
            let kv = Matrix::new(n, 1, k.clone());
            assert_eq!(a.mul(&kv), Matrix::zeros(a.rows(), 1), "A·k ≠ 0");
            // A kernel basis vector must be nonzero (primitive lattice basis).
            assert!(k.iter().any(|e| !e.is_zero()), "zero kernel vector");
        }
    }

    #[test]
    fn hnf_known_small() {
        // Hand example: two generators of a rank-1 row lattice in ℤ².
        let a = Matrix::from_rows(alloc::vec![
            alloc::vec![Int::from_i64(2), Int::from_i64(4)],
            alloc::vec![Int::from_i64(3), Int::from_i64(6)],
        ]);
        let h = a.hermite_normal_form();
        // Row lattice is spanned by (1,2); HNF = [[1,2],[0,0]].
        assert_eq!(
            h,
            Matrix::from_rows(alloc::vec![
                alloc::vec![Int::ONE, Int::from_i64(2)],
                alloc::vec![Int::ZERO, Int::ZERO],
            ])
        );
    }

    #[test]
    fn snf_known_example() {
        // Classic hand-checkable case: invariant factors (2, 2, 156).
        let a = Matrix::from_rows(alloc::vec![
            alloc::vec![Int::from_i64(2), Int::from_i64(4), Int::from_i64(4)],
            alloc::vec![Int::from_i64(-6), Int::from_i64(6), Int::from_i64(12)],
            alloc::vec![Int::from_i64(10), Int::from_i64(4), Int::from_i64(16)],
        ]);
        assert_eq!(
            a.invariant_factors(),
            alloc::vec![Int::from_i64(2), Int::from_i64(2), Int::from_i64(156)]
        );
        assert_eq!(a.determinant().abs(), Int::from_i64(624));
        assert_snf_contract(&a);
    }

    #[test]
    fn structured_cases() {
        let mut rng = Lcg::new(0x0BAD_F00D);
        let cases = alloc::vec![
            // identity, zero, single row/col, rank-deficient, zero row/col.
            Matrix::<Int>::identity(4),
            Matrix::<Int>::zeros(3, 4),
            Matrix::from_rows(alloc::vec![alloc::vec![
                Int::from_i64(6),
                Int::from_i64(10),
                Int::from_i64(15)
            ]]),
            Matrix::from_rows(alloc::vec![
                alloc::vec![Int::from_i64(2)],
                alloc::vec![Int::from_i64(3)],
                alloc::vec![Int::from_i64(5)],
            ]),
            // rank-2, 3×3 (third row = row1 + row2).
            Matrix::from_rows(alloc::vec![
                alloc::vec![Int::from_i64(1), Int::from_i64(2), Int::from_i64(3)],
                alloc::vec![Int::from_i64(4), Int::from_i64(5), Int::from_i64(6)],
                alloc::vec![Int::from_i64(5), Int::from_i64(7), Int::from_i64(9)],
            ]),
            // a matrix with a zero column and a zero row.
            Matrix::from_rows(alloc::vec![
                alloc::vec![Int::from_i64(3), Int::ZERO, Int::from_i64(9)],
                alloc::vec![Int::from_i64(6), Int::ZERO, Int::from_i64(3)],
                alloc::vec![Int::ZERO, Int::ZERO, Int::ZERO],
            ]),
        ];
        for a in &cases {
            assert_hnf_contract(a, &mut rng);
            assert_snf_contract(a);
            assert_kernel(a);
        }
    }

    #[test]
    fn random_cases() {
        let mut rng = Lcg::new(0x5EED_1234);
        let shapes = [
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
            (2, 4),
            (4, 2),
            (3, 5),
            (5, 3),
            (6, 6),
        ];
        for &(m, n) in &shapes {
            for _ in 0..6 {
                let a = rng.imatrix(m, n, 6);
                assert_hnf_contract(&a, &mut rng);
                assert_snf_contract(&a);
                assert_kernel(&a);
            }
        }
    }

    #[test]
    fn rank_deficient_and_larger_entries() {
        let mut rng = Lcg::new(0xDEAD_BEEF);
        // Force rank deficiency by making some rows integer combinations of others.
        for _ in 0..8 {
            let base = rng.imatrix(2, 5, 20);
            let mut rows: Vec<Vec<Int>> = (0..2)
                .map(|i| (0..5).map(|j| base.get(i, j).clone()).collect())
                .collect();
            // third row = 2·row0 − 3·row1, fourth row = row0 + row1.
            let combo1: Vec<Int> = (0..5)
                .map(|j| {
                    Int::from_i64(2)
                        .mul(base.get(0, j))
                        .sub(&Int::from_i64(3).mul(base.get(1, j)))
                })
                .collect();
            let combo2: Vec<Int> = (0..5).map(|j| base.get(0, j).add(base.get(1, j))).collect();
            rows.push(combo1);
            rows.push(combo2);
            let a = Matrix::from_rows(rows);
            assert!(a.rank() <= 2);
            assert_hnf_contract(&a, &mut rng);
            assert_snf_contract(&a);
            assert_kernel(&a);
        }
    }

    #[test]
    fn image_basis_spans_columns() {
        // The image lattice basis must contain every original column, and its
        // size must equal the rank.
        let mut rng = Lcg::new(0x1CE_B00C);
        for _ in 0..6 {
            let a = rng.imatrix(4, 3, 6);
            let basis = a.image_basis();
            assert_eq!(basis.len(), a.rank());
            let m = a.rows();
            // Each column of A must be an integer combination of the basis.
            let bcols = basis.len();
            let mut bdata = alloc::vec![Int::ZERO; m * bcols];
            for (jb, bv) in basis.iter().enumerate() {
                for (i, e) in bv.iter().enumerate() {
                    bdata[i * bcols + jb] = e.clone();
                }
            }
            let bmat = Matrix::new(m, bcols, bdata);
            for j in 0..a.cols() {
                let col: Vec<Int> = (0..m).map(|i| a.get(i, j).clone()).collect();
                assert!(
                    bmat.solve_integer(&col).is_some(),
                    "column {j} not in image lattice"
                );
            }
        }
    }

    #[test]
    fn solve_integer_roundtrip_and_inconsistency() {
        let mut rng = Lcg::new(0xF00D_F00D);
        for _ in 0..10 {
            let a = rng.imatrix(4, 4, 5);
            // Solvable system: pick x0, set b = A·x0.
            let x0: Vec<Int> = (0..4).map(|_| rng.small(4)).collect();
            let xv = Matrix::new(4, 1, x0.clone());
            let b: Vec<Int> = (0..4).map(|i| a.mul(&xv).get(i, 0).clone()).collect();
            match a.solve_integer(&b) {
                Some(x) => {
                    let xm = Matrix::new(4, 1, x);
                    let bcheck: Vec<Int> = (0..4).map(|i| a.mul(&xm).get(i, 0).clone()).collect();
                    assert_eq!(bcheck, b, "A·x ≠ b");
                }
                None => panic!("consistent system reported unsolvable"),
            }
        }
        // A clearly inconsistent system: 2x = 1 has no integer solution.
        let a = Matrix::from_rows(alloc::vec![alloc::vec![Int::from_i64(2)]]);
        assert!(a.solve_integer(&[Int::ONE]).is_none());
        // Rank-deficient inconsistency: rows (1,1) and (1,1) with b = (0,1).
        let a = Matrix::from_rows(alloc::vec![
            alloc::vec![Int::ONE, Int::ONE],
            alloc::vec![Int::ONE, Int::ONE],
        ]);
        assert!(a.solve_integer(&[Int::ZERO, Int::ONE]).is_none());
        assert!(
            a.solve_integer(&[Int::from_i64(2), Int::from_i64(2)])
                .is_some()
        );
    }

    #[test]
    fn empty_and_degenerate_shapes() {
        // Zero matrix: everything is trivial/consistent.
        let z = Matrix::<Int>::zeros(3, 3);
        assert_eq!(z.rank(), 0);
        assert!(z.invariant_factors().is_empty());
        assert_eq!(z.kernel().len(), 3);
        assert_eq!(z.image_basis().len(), 0);
        assert_eq!(z.solve_integer(&zero_vec(3)), Some(zero_vec(3)));
        assert!(z.solve_integer(&[Int::ZERO, Int::ZERO, Int::ONE]).is_none());
    }

    /// The modulo-determinant HNF (both branches) must equal the reference
    /// extended-gcd reduction **bit-for-bit** on full-rank inputs.
    #[cfg(feature = "rational")]
    #[test]
    fn modular_hnf_matches_reference() {
        let mut rng = Lcg::new(0xC0FF_EE42);
        // Square (nonsingular) and wide (full row rank), assorted sizes/entry
        // sizes straddling the routing threshold.
        let cases: &[(usize, usize)] = &[
            (2, 2),
            (3, 3),
            (5, 5),
            (6, 6),
            (7, 7),
            (8, 8),
            (2, 4),
            (3, 6),
            (5, 8),
            (6, 12),
            (4, 5),
        ];
        for &(m, n) in cases {
            for &bits in &[4u32, 40, 80, 200] {
                for _ in 0..3 {
                    let a = rng.full_rank(m, n, bits);
                    let reference = Matrix::new(m, n, a.hnf_impl(false).0);
                    // Force the modular path regardless of the size threshold.
                    let modular = if m == n {
                        Matrix::new(m, n, hnf_mod_det(&a.data, n, &a.determinant().abs()))
                    } else {
                        Matrix::new(m, n, a.hnf_modular_wide().expect("full row rank"))
                    };
                    assert_eq!(
                        modular, reference,
                        "modular HNF ≠ reference ({m}×{n}, {bits}b)"
                    );
                    // Canonical shape + ∏pivots = |det| are checked by the shared
                    // helper (also exercises the public routed path).
                    check_hnf_shape(&modular);
                }
            }
        }
    }

    /// The routed public `hermite_normal_form` (which dispatches to the modular
    /// path for large full-rank inputs) still equals the reference and satisfies
    /// the full HNF contract.
    #[test]
    fn routed_hnf_large_entries_contract() {
        let mut rng = Lcg::new(0x1234_ABCD);
        for &(m, n) in &[(6usize, 6usize), (8, 8), (6, 10)] {
            let a = rng.full_rank(m, n, 300);
            let reference = Matrix::new(m, n, a.hnf_impl(false).0);
            assert_eq!(a.hermite_normal_form(), reference, "routed HNF ≠ reference");
            assert_hnf_contract(&a, &mut rng);
        }
    }

    #[test]
    #[ignore = "slow: larger random matrices"]
    fn large_random_stress() {
        let mut rng = Lcg::new(0x00A1_1CE5);
        for &(m, n) in &[(10usize, 10usize), (12, 8), (8, 12), (14, 14)] {
            for _ in 0..4 {
                let a = rng.imatrix(m, n, 30);
                assert_hnf_contract(&a, &mut rng);
                assert_snf_contract(&a);
                assert_kernel(&a);
            }
        }
    }

    /// Exercises the *routed* public path at and above the routing threshold
    /// (`n ≥ 20`, including the wide branch) and checks it is bit-identical to the
    /// reference reduction. Kept `#[ignore]` because the reference side is slow at
    /// these sizes.
    #[cfg(feature = "rational")]
    #[test]
    #[ignore = "slow: reference reduction at n ≥ 20"]
    fn routed_hnf_threshold_matches_reference() {
        let mut rng = Lcg::new(0x2468_ACE0);
        for &(m, n) in &[(20usize, 20usize), (22, 22), (20, 40), (24, 48)] {
            let a = rng.full_rank(m, n, 96);
            assert!(hnf_modular_worthwhile(&a), "expected routing at {m}×{n}");
            let reference = Matrix::new(m, n, a.hnf_impl(false).0);
            assert_eq!(
                a.hermite_normal_form(),
                reference,
                "routed HNF ≠ reference ({m}×{n})"
            );
        }
    }

    /// Head-to-head timing of the reference extended-gcd reduction against the
    /// modulo-determinant path, over full-rank square and `n×2n` matrices at
    /// assorted dimensions and entry sizes. Run with:
    /// `cargo test --release --features matrix,int,rational -- --ignored --nocapture bench_modular_vs_reference`
    #[cfg(feature = "rational")]
    #[test]
    #[ignore = "benchmark: run in --release with --nocapture"]
    fn bench_modular_vs_reference() {
        use std::time::Instant;
        fn secs<F: FnMut()>(mut f: F) -> f64 {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64()
        }
        let mut rng = Lcg::new(0xB0A7_1235);
        std::println!("shape        n  bits   reference     modular   speedup");
        for &(square, dims) in &[(true, &[8usize, 16, 24][..]), (false, &[8, 16, 24][..])] {
            for &n in dims {
                let cols = if square { n } else { 2 * n };
                for &bits in &[32u32, 128, 512] {
                    let a = rng.full_rank(n, cols, bits);
                    let t_ref = secs(|| {
                        let _ = a.hnf_impl(false).0;
                    });
                    // Force the modular path (includes its determinant / pivot pass).
                    let t_mod = secs(|| {
                        let _ = if square {
                            Matrix::new(n, cols, hnf_mod_det(&a.data, n, &a.determinant().abs()))
                        } else {
                            Matrix::new(n, cols, a.hnf_modular_wide().expect("full row rank"))
                        };
                    });
                    let shape = if square { "square" } else { "n x 2n" };
                    std::println!(
                        "{shape}  {n:>3} {bits:>5} {t_ref:>10.4}s {t_mod:>10.4}s  {:>6.1}x",
                        t_ref / t_mod
                    );
                }
            }
        }
    }
}
