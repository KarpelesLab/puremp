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
