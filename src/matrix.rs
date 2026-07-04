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

use crate::ring::Ring;
use alloc::vec::Vec;
use core::fmt;

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
    pub fn mul(&self, rhs: &Matrix<T>) -> Matrix<T> {
        assert_eq!(self.cols, rhs.rows, "Matrix::mul: inner dimension mismatch");
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
