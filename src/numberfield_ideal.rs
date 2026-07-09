//! The ring of integers `O_K`, its ideals, and the factorization of rational
//! primes for a [`NumberField`] `K = ℚ(θ) = ℚ[x]/(T)`.
//!
//! This is phase 2 of the number-field subsystem, built on top of
//! [`crate::numberfield`]. Everything here assumes the defining polynomial `T`
//! is **monic with integer coefficients**, so that `θ` is an algebraic integer
//! and `ℤ[θ] ⊆ O_K`.
//!
//! # What is provided
//!
//! - [`Order`] — the **maximal order** `O_K` (ring of integers), computed by the
//!   Round 2 / Pohst–Zassenhaus algorithm (Cohen, *A Course in Computational
//!   Algebraic Number Theory* (CCANT), Algorithm 6.1.8). An order is stored by an
//!   **integral basis**: an `n × n` matrix over ℚ whose columns are the
//!   power-basis coordinates of `ω₀, …, ω_{n−1}` (with a cleared common
//!   denominator, in Hermite normal form). The **field discriminant** `d_K`
//!   satisfies `disc(T) = [O_K : ℤ[θ]]² · d_K`.
//! - [`Ideal`] — an ideal of `O_K`, stored by the HNF ℤ-basis of its elements
//!   (an `n × n` integer matrix in terms of the integral basis of `O_K`), with
//!   multiplication, sum, norm (`= [O_K : 𝔞]`), containment, and equality.
//! - [`PrimeIdeal`] — a prime ideal above a rational prime `p`, carrying `p`, the
//!   residue degree `f`, the ramification index `e`, and its ideal basis.
//! - [`Order::factor_prime`] — the decomposition `pO_K = ∏ 𝔭ᵢ^{eᵢ}`.
//!
//! # Algorithms (clean-room, from the open literature)
//!
//! **Maximal order (Round 2).** Starting from `ℤ[θ]`, for every prime `p` with
//! `p² | disc(T)` we make the order `p`-maximal: iterate the *ring of
//! multipliers* `O' = (I_p : I_p)` of the `p`-radical `I_p` (the radical of
//! `pO`) until it stabilises (CCANT §6.1). The `p`-radical is the kernel of the
//! `𝔽_p`-linear Frobenius power `x ↦ x^{p^k}` on `O/pO` (`p^k ≥ n`); the ring of
//! multipliers is computed as a lattice `{x ∈ K : x I_p ⊆ I_p}` via the Smith
//! normal form.
//!
//! **Ideal multiplication.** `𝔞·𝔟` is the ℤ-span of all `n²` products of the two
//! bases, reduced back to `n` generators by Hermite normal form.
//!
//! **Prime factorization.** For `p ∤ [O_K:ℤ[θ]]` we use **Kummer–Dedekind**:
//! factor `T ≡ ∏ ḡᵢ^{eᵢ} (mod p)` over `GF(p)` and set `𝔭ᵢ = (p, gᵢ(θ))`,
//! `fᵢ = deg gᵢ`. For `p | [O_K:ℤ[θ]]` we split the finite `𝔽_p`-algebra
//! `O_K/pO_K` into its local factors with a Berlekamp-style idempotent search
//! (CCANT §6.2), reading off `𝔭ᵢ`, `fᵢ` from the components and `eᵢ` from the
//! `𝔭ᵢ`-adic valuation of `pO_K`.

// This module is dense exact linear algebra over explicit `i×j` index ranges
// (structure constants, Frobenius/idempotent matrices); range-indexed loops read
// far closer to the underlying mathematics than iterator adapters here.
#![allow(clippy::needless_range_loop)]

use alloc::vec::Vec;
use core::fmt;

use crate::int::Int;
use crate::matrix::Matrix;
use crate::numberfield::NumberField;
use crate::poly::Poly;
use crate::random::SeedRng;
use crate::rational::Rational;

// ===========================================================================
// Small GF(p) linear algebra over `Vec<Vec<Int>>` (entries kept in `[0, p)`).
// ===========================================================================

/// Reduces `a` into `[0, p)`.
fn gmod(a: &Int, p: &Int) -> Int {
    a.rem_euclid(p)
}

/// Row-reduces `rows` (an `m × ncols` matrix over `GF(p)`) to reduced row
/// echelon form in place, returning `(reduced_nonzero_rows, pivot_columns)`.
/// Pivot entries are normalised to `1`.
fn rref_modp(rows: &[Vec<Int>], ncols: usize, p: &Int) -> (Vec<Vec<Int>>, Vec<usize>) {
    let mut m: Vec<Vec<Int>> = rows
        .iter()
        .map(|r| r.iter().map(|x| gmod(x, p)).collect())
        .collect();
    let mut pivots: Vec<usize> = Vec::new();
    let mut r = 0usize;
    for col in 0..ncols {
        // Find a pivot in this column at or below row r.
        let piv = (r..m.len()).find(|&i| !m[i][col].is_zero());
        let piv = match piv {
            Some(pv) => pv,
            None => continue,
        };
        m.swap(r, piv);
        // Normalise pivot row to leading 1.
        let inv = m[r][col].modinv(p).expect("pivot invertible mod prime");
        for c in 0..ncols {
            m[r][c] = gmod(&m[r][c].mul(&inv), p);
        }
        // Eliminate the column from every other row.
        for i in 0..m.len() {
            if i == r || m[i][col].is_zero() {
                continue;
            }
            let f = m[i][col].clone();
            for c in 0..ncols {
                m[i][c] = gmod(&m[i][c].sub(&f.mul(&m[r][c])), p);
            }
        }
        pivots.push(col);
        r += 1;
        if r == m.len() {
            break;
        }
    }
    m.truncate(r);
    (m, pivots)
}

/// A `GF(p)` basis of the null space `{x : M x = 0}` of the `m × ncols` matrix
/// `rows`, as column vectors of length `ncols`.
fn kernel_modp(rows: &[Vec<Int>], ncols: usize, p: &Int) -> Vec<Vec<Int>> {
    let (red, pivots) = rref_modp(rows, ncols, p);
    let is_pivot = |c: usize| pivots.contains(&c);
    let mut basis = Vec::new();
    for free in 0..ncols {
        if is_pivot(free) {
            continue;
        }
        let mut v = alloc::vec![Int::ZERO; ncols];
        v[free] = Int::ONE;
        for (row, &pc) in red.iter().zip(&pivots) {
            // pivot variable pc = -sum(free entries); here only `free` is set.
            v[pc] = gmod(&row[free].neg(), p);
        }
        basis.push(v);
    }
    basis
}

/// The rank over `GF(p)`.
fn rank_modp(rows: &[Vec<Int>], ncols: usize, p: &Int) -> usize {
    rref_modp(rows, ncols, p).1.len()
}

/// `A · B` over `GF(p)` for square `n × n` matrices stored row-major as
/// `Vec<Vec<Int>>`.
fn matmul_modp(a: &[Vec<Int>], b: &[Vec<Int>], n: usize, p: &Int) -> Vec<Vec<Int>> {
    let mut out = alloc::vec![alloc::vec![Int::ZERO; n]; n];
    for i in 0..n {
        for k in 0..n {
            if a[i][k].is_zero() {
                continue;
            }
            for j in 0..n {
                out[i][j] = out[i][j].add(&a[i][k].mul(&b[k][j]));
            }
        }
    }
    for row in out.iter_mut() {
        for v in row.iter_mut() {
            *v = gmod(v, p);
        }
    }
    out
}

/// `M^k` over `GF(p)` (`k ≥ 0`, `k = 0` giving the identity).
fn matpow_modp(m: &[Vec<Int>], k: usize, n: usize, p: &Int) -> Vec<Vec<Int>> {
    let mut result: Vec<Vec<Int>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| if i == j { Int::ONE } else { Int::ZERO })
                .collect()
        })
        .collect();
    for _ in 0..k {
        result = matmul_modp(&result, m, n, p);
    }
    result
}

// ===========================================================================
// GF(p) algebra element arithmetic (for the p | index prime splitting).
// ===========================================================================

/// Product of two algebra elements `x, y` (length-`n` `GF(p)` vectors) using the
/// structure constants `table[a][b] = ω_a·ω_b` (in `O`-coordinates, mod `p`).
fn algebra_mul(table: &[Vec<Vec<Int>>], x: &[Int], y: &[Int], n: usize, p: &Int) -> Vec<Int> {
    let mut acc = alloc::vec![Int::ZERO; n];
    for a in 0..n {
        if x[a].is_zero() {
            continue;
        }
        for b in 0..n {
            if y[b].is_zero() {
                continue;
            }
            let coef = gmod(&x[a].mul(&y[b]), p);
            let t = &table[a][b];
            for i in 0..n {
                acc[i] = acc[i].add(&coef.mul(&t[i]));
            }
        }
    }
    acc.iter().map(|v| gmod(v, p)).collect()
}

/// Vector subtraction mod `p`.
fn vec_sub_modp(a: &[Int], b: &[Int], p: &Int) -> Vec<Int> {
    a.iter().zip(b).map(|(x, y)| gmod(&x.sub(y), p)).collect()
}

/// Scalar multiple `s · v` mod `p`.
fn vec_scale_modp(s: &Int, v: &[Int], p: &Int) -> Vec<Int> {
    v.iter().map(|x| gmod(&s.mul(x), p)).collect()
}

// ===========================================================================
// Integer-matrix helpers.
// ===========================================================================

/// Builds an `m × n` integer matrix from a list of rows.
fn imat_from_rows(rows: &[Vec<Int>], n: usize) -> Matrix<Int> {
    let m = rows.len();
    let mut data = Vec::with_capacity(m * n);
    for r in rows {
        debug_assert_eq!(r.len(), n);
        data.extend(r.iter().cloned());
    }
    Matrix::new(m, n, data)
}

/// Row Hermite normal form of `rows`, keeping the top `n` (pivot) rows — a
/// canonical ℤ-basis of the row lattice, assumed to be full rank `n`.
fn hnf_top_n(rows: &[Vec<Int>], n: usize) -> Matrix<Int> {
    let h = imat_from_rows(rows, n).hermite_normal_form();
    let mut data = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            data.push(h.get(i, j).clone());
        }
    }
    Matrix::new(n, n, data)
}

/// Converts an integer matrix to a rational one.
fn int_to_rat(m: &Matrix<Int>) -> Matrix<Rational> {
    let r = m.rows();
    let c = m.cols();
    let data: Vec<Rational> = m
        .as_slice()
        .iter()
        .map(|x| Rational::from_integer(x.clone()))
        .collect();
    Matrix::new(r, c, data)
}

/// Whether every entry of a rational matrix is an integer.
fn is_integer_matrix(m: &Matrix<Rational>) -> bool {
    m.as_slice().iter().all(|x| x.is_integer())
}

/// The first `n` coefficients of a rational polynomial, treating the zero
/// polynomial (and any missing high coefficient) as zero.
fn poly_coords(p: &Poly<Rational>, n: usize) -> Vec<Rational> {
    if p.is_zero() {
        return alloc::vec![Rational::ZERO; n];
    }
    (0..n).map(|i| p.coeff(i)).collect()
}

// ===========================================================================
// Order (ring of integers).
// ===========================================================================

/// An **order** in a number field — in particular the maximal order `O_K`
/// (ring of integers), returned by [`NumberField::maximal_order`].
///
/// The order is stored by its **integral basis**: an `n × n` matrix over ℚ whose
/// column `j` holds the power-basis coordinates of the `j`-th basis element
/// `ω_j = Σ_i basis[i][j] · θ^i`. For the maximal order this basis is in Hermite
/// normal form with a cleared common denominator.
#[derive(Clone)]
pub struct Order {
    field: NumberField,
    n: usize,
    /// Column `j` = power-basis coordinates of `ω_j`.
    basis: Matrix<Rational>,
    /// Inverse of `basis` (power coordinates → order coordinates).
    basis_inv: Matrix<Rational>,
    /// `disc(T)`, the discriminant of the defining polynomial.
    disc_t: Int,
    /// `[O_K : ℤ[θ]]` (for the maximal order); `1` for `ℤ[θ]` itself.
    index: Int,
    /// The field discriminant `d_K = disc(T)/index²`.
    d_k: Int,
}

impl fmt::Debug for Order {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Order(degree {}, index {}, d_K {})",
            self.n, self.index, self.d_k
        )
    }
}

impl Order {
    /// The number field this order lives in.
    pub fn field(&self) -> NumberField {
        self.field.clone()
    }

    /// The degree `n = [K:ℚ]`.
    #[inline]
    pub fn degree(&self) -> usize {
        self.n
    }

    /// The field discriminant `d_K`.
    #[inline]
    pub fn discriminant(&self) -> Int {
        self.d_k.clone()
    }

    /// The index `[O_K : ℤ[θ]]`, where `disc(T) = index² · d_K`.
    #[inline]
    pub fn index(&self) -> Int {
        self.index.clone()
    }

    /// The integral basis as an `n × n` matrix over ℚ (column `j` = power-basis
    /// coordinates of `ω_j`).
    pub fn integral_basis(&self) -> Matrix<Rational> {
        self.basis.clone()
    }

    // --- coordinate conversions ---

    /// power-basis coordinates `v` of an element given by order coordinates `c`
    /// (`v = basis · c`).
    fn order_to_power(&self, c: &[Rational]) -> Vec<Rational> {
        let n = self.n;
        (0..n)
            .map(|i| {
                let mut acc = Rational::ZERO;
                for j in 0..n {
                    acc = acc.add(&self.basis.get(i, j).mul(&c[j]));
                }
                acc
            })
            .collect()
    }

    /// order coordinates `c` of an element given by power-basis coordinates `v`
    /// (`c = basis_inv · v`).
    fn power_to_order(&self, v: &[Rational]) -> Vec<Rational> {
        let n = self.n;
        (0..n)
            .map(|i| {
                let mut acc = Rational::ZERO;
                for j in 0..n {
                    acc = acc.add(&self.basis_inv.get(i, j).mul(&v[j]));
                }
                acc
            })
            .collect()
    }

    /// Product of two elements given in order coordinates (rational).
    fn mul_order_rat(&self, a: &[Rational], b: &[Rational]) -> Vec<Rational> {
        let va = self.order_to_power(a);
        let vb = self.order_to_power(b);
        let ea = self.field.element(Poly::new(va));
        let eb = self.field.element(Poly::new(vb));
        let ec = ea.mul(&eb);
        let vc = poly_coords(ec.poly(), self.n);
        self.power_to_order(&vc)
    }

    /// Product of two order elements given in integer order coordinates.
    fn mul_order_int(&self, a: &[Int], b: &[Int]) -> Vec<Int> {
        let ar: Vec<Rational> = a
            .iter()
            .map(|x| Rational::from_integer(x.clone()))
            .collect();
        let br: Vec<Rational> = b
            .iter()
            .map(|x| Rational::from_integer(x.clone()))
            .collect();
        self.mul_order_rat(&ar, &br)
            .iter()
            .map(|x| {
                x.to_integer()
                    .expect("product of order elements is integral")
            })
            .collect()
    }

    /// The `n × n` integer matrix of "multiply by `alpha`" on `O` (column `k` =
    /// order coordinates of `alpha · ω_k`), for `alpha` in integer order
    /// coordinates.
    fn mul_matrix_int(&self, alpha: &[Int]) -> Matrix<Int> {
        let n = self.n;
        let mut data = alloc::vec![Int::ZERO; n * n];
        for k in 0..n {
            let mut ek = alloc::vec![Int::ZERO; n];
            ek[k] = Int::ONE;
            let col = self.mul_order_int(alpha, &ek);
            for (i, ci) in col.into_iter().enumerate() {
                data[i * n + k] = ci;
            }
        }
        Matrix::new(n, n, data)
    }

    /// The order coordinates of `1 ∈ O` (integer vector).
    fn one_coords(&self) -> Vec<Int> {
        let n = self.n;
        let mut v = alloc::vec![Rational::ZERO; n];
        v[0] = Rational::ONE; // power coordinates of 1
        self.power_to_order(&v)
            .iter()
            .map(|x| x.to_integer().expect("1 lies in the order"))
            .collect()
    }

    // --- Round 2 building blocks ---

    /// The `n × n` Frobenius matrix `F` over `GF(p)`: column `k` holds the
    /// order coordinates (mod `p`) of `ω_k^p`.
    fn frobenius_modp(&self, p: &Int) -> Vec<Vec<Int>> {
        let n = self.n;
        let pe = p.to_i64().expect("prime characteristic fits in i64");
        // f[i][k]
        let mut f = alloc::vec![alloc::vec![Int::ZERO; n]; n];
        for k in 0..n {
            let mut ek = alloc::vec![Rational::ZERO; n];
            ek[k] = Rational::ONE;
            let vpow = self.order_to_power(&ek);
            let elt = self.field.element(Poly::new(vpow)).pow(pe);
            let vc = poly_coords(elt.poly(), n);
            let coords = self.power_to_order(&vc);
            for (i, c) in coords.iter().enumerate() {
                let ci = c.to_integer().expect("ω_k^p lies in the order");
                f[i][k] = gmod(&ci, p);
            }
        }
        f
    }

    /// A `GF(p)`-basis of the `p`-radical modulo `pO` — the kernel of
    /// `x ↦ x^{p^k}` on `O/pO` with `p^k ≥ n`.
    fn radical_modp(&self, p: &Int) -> Vec<Vec<Int>> {
        let n = self.n;
        let f = self.frobenius_modp(p);
        // smallest k with p^k >= n
        let mut k = 0usize;
        let mut acc = Int::ONE;
        while acc < Int::from_i64(n as i64) {
            acc = acc.mul(p);
            k += 1;
        }
        let fk = matpow_modp(&f, k, n, p);
        kernel_modp(&fk, n, p)
    }

    /// The `p`-radical `I_p` as a full-rank integer lattice (columns = ℤ-basis in
    /// order coordinates): the lattice generated by the radical's `GF(p)`-basis
    /// (lifted) together with `p·O`.
    fn p_radical_lattice(&self, p: &Int) -> Matrix<Int> {
        let n = self.n;
        let ker = self.radical_modp(p);
        let mut rows: Vec<Vec<Int>> = ker;
        for i in 0..n {
            let mut e = alloc::vec![Int::ZERO; n];
            e[i] = p.clone();
            rows.push(e);
        }
        // hnf_top_n gives basis as rows; transpose to columns.
        hnf_top_n(&rows, n).transpose()
    }

    /// The ring of multipliers `(I : I)` of an ideal `I` given by the integer
    /// matrix `w` (columns = ℤ-basis of `I` in order coordinates), returned as a
    /// new [`Order`]. Uses the lattice `{x ∈ K : x I ⊆ I}` computed via the Smith
    /// normal form.
    fn multiplier_ring(&self, w: &Matrix<Int>) -> Order {
        let n = self.n;
        let det = w.determinant();
        let det_abs = det.abs();
        let w_rat = int_to_rat(w);
        let w_inv = w_rat.inverse().expect("radical lattice is full rank");
        // adj = det · W⁻¹ (integer matrix).
        let det_rat = Rational::from_integer(det.clone());
        let adj_rat = w_inv.scalar_mul(&det_rat);
        let adj = {
            let data: Vec<Int> = adj_rat
                .as_slice()
                .iter()
                .map(|x| x.to_integer().expect("adjugate is integral"))
                .collect();
            Matrix::new(n, n, data)
        };
        // Stack G_j = adj · M_j (M_j = mult-by-(column j of W)).
        let mut g_rows: Vec<Vec<Int>> = Vec::with_capacity(n * n);
        for j in 0..n {
            let aj: Vec<Int> = (0..n).map(|i| w.get(i, j).clone()).collect();
            let mj = self.mul_matrix_int(&aj);
            let gj = adj.mul(&mj);
            for r in 0..n {
                g_rows.push((0..n).map(|c| gj.get(r, c).clone()).collect());
            }
        }
        let g = imat_from_rows(&g_rows, n);
        // Smith form: S = U·G·V. c ∈ L ⟺ Gc ∈ det_abs·ℤ ⟺ (V⁻¹c)_i ∈ (det_abs/s_i)ℤ.
        let (_u, s, v) = g.smith_normal_form_with_transforms();
        // L-basis column i = (det_abs / s_i) · V[:,i]  (order coordinates).
        let mut lbasis = Matrix::<Rational>::zeros(n, n);
        for i in 0..n {
            let si = s.get(i, i).clone();
            let scale = Rational::new(det_abs.clone(), si);
            for r in 0..n {
                let vr = Rational::from_integer(v.get(r, i).clone());
                lbasis.set(r, i, scale.mul(&vr));
            }
        }
        // New order's power-basis matrix = self.basis · Lbasis.
        let new_basis = self.basis.mul(&lbasis);
        let new_inv = new_basis.inverse().expect("multiplier ring is full rank");
        Order {
            field: self.field.clone(),
            n,
            basis: new_basis,
            basis_inv: new_inv,
            disc_t: self.disc_t.clone(),
            index: Int::ONE,
            d_k: Int::ZERO,
        }
    }

    /// Makes the order `p`-maximal by iterating the ring of multipliers of the
    /// `p`-radical until it stabilises.
    fn make_p_maximal(&self, p: &Int) -> Order {
        let mut cur = self.clone();
        loop {
            let radical = cur.p_radical_lattice(p);
            let next = cur.multiplier_ring(&radical);
            // next ⊇ cur always; equal ⟺ cur⁻¹·next is integral.
            let change = mat_mul_rat(&cur.basis_inv, &next.basis);
            if is_integer_matrix(&change) {
                return cur;
            }
            cur = next;
        }
    }

    /// Canonicalises the integral basis to Hermite normal form with a cleared
    /// common denominator, and fills in `index` and `d_k`.
    fn finalize(mut self) -> Order {
        let n = self.n;
        // Common denominator of all basis entries.
        let mut den = Int::ONE;
        for x in self.basis.as_slice() {
            den = den.lcm(x.denominator());
        }
        // Integer matrix = den · basis (columns = den·ω_j power coordinates).
        let mut bint_rows: Vec<Vec<Int>> = Vec::with_capacity(n);
        for i in 0..n {
            let row: Vec<Int> = (0..n)
                .map(|j| {
                    let e = self.basis.get(i, j);
                    // den · e is integral since den is a common multiple.
                    e.numerator().mul(&den.div_exact(e.denominator()))
                })
                .collect();
            bint_rows.push(row);
        }
        // Column HNF: HNF of the transpose (row lattice = column lattice), then
        // transpose the top n rows back to columns.
        let bint = imat_from_rows(&bint_rows, n); // rows i, cols j: den·basis[i][j]
        let cols_as_rows: Vec<Vec<Int>> = (0..n)
            .map(|j| (0..n).map(|i| bint.get(i, j).clone()).collect())
            .collect();
        let h = hnf_top_n(&cols_as_rows, n); // rows = HNF basis of column lattice
        let h_cols = h.transpose(); // columns = basis vectors
        // New rational basis = (1/den) · h_cols.
        let mut new_basis = Matrix::<Rational>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                new_basis.set(i, j, Rational::new(h_cols.get(i, j).clone(), den.clone()));
            }
        }
        let new_inv = new_basis.inverse().expect("order basis is full rank");
        // index = 1/|det(basis)|.
        let detb = new_basis.determinant();
        let index = detb
            .abs()
            .recip()
            .to_integer()
            .expect("index [O_K:ℤ[θ]] is an integer");
        let d_k = self.disc_t.div_exact(&index.square());
        self.basis = new_basis;
        self.basis_inv = new_inv;
        self.index = index;
        self.d_k = d_k;
        self
    }

    // --- ideal constructors ---

    /// The unit ideal `O_K` itself (the whole ring).
    pub fn unit_ideal(&self) -> Ideal {
        let n = self.n;
        let basis = Matrix::<Int>::identity(n);
        Ideal {
            order: self.clone(),
            n,
            basis,
        }
    }

    /// The principal ideal `(α)` generated by an algebraic integer `α`, given by
    /// its **power-basis** coordinates (which must be integral in `O_K`).
    pub fn principal_ideal(&self, alpha_power: &[Rational]) -> Ideal {
        let coords = self.power_to_order(alpha_power);
        let alpha: Vec<Int> = coords
            .iter()
            .map(|x| {
                x.to_integer()
                    .expect("α must be an algebraic integer in O_K")
            })
            .collect();
        self.principal_from_order_coords(&alpha)
    }

    /// The principal ideal `(α)` for `α` given in integer order coordinates.
    fn principal_from_order_coords(&self, alpha: &[Int]) -> Ideal {
        let n = self.n;
        let mut rows: Vec<Vec<Int>> = Vec::with_capacity(n);
        for k in 0..n {
            let mut ek = alloc::vec![Int::ZERO; n];
            ek[k] = Int::ONE;
            rows.push(self.mul_order_int(alpha, &ek));
        }
        Ideal {
            order: self.clone(),
            n,
            basis: hnf_top_n(&rows, n),
        }
    }

    /// The ideal generated by `p` and the element `beta` (integer order
    /// coordinates): `(p, β) = pO + βO`.
    fn ideal_p_beta(&self, p: &Int, beta: &[Int]) -> Ideal {
        let n = self.n;
        let mut rows: Vec<Vec<Int>> = Vec::new();
        for k in 0..n {
            let mut ek = alloc::vec![Int::ZERO; n];
            ek[k] = p.clone();
            rows.push(ek);
        }
        for k in 0..n {
            let mut ek = alloc::vec![Int::ZERO; n];
            ek[k] = Int::ONE;
            rows.push(self.mul_order_int(beta, &ek));
        }
        Ideal {
            order: self.clone(),
            n,
            basis: hnf_top_n(&rows, n),
        }
    }

    // --- prime factorization ---

    /// Factors the rational prime `p` in `O_K` as `pO_K = ∏ 𝔭ᵢ^{eᵢ}`, returning
    /// each prime ideal together with its ramification index `eᵢ`.
    ///
    /// Uses **Kummer–Dedekind** when `p ∤ [O_K:ℤ[θ]]` and the general
    /// `O_K/pO_K`-splitting otherwise.
    ///
    /// # Panics
    /// If `p` is not a prime.
    pub fn factor_prime(&self, p: &Int) -> Vec<(PrimeIdeal, usize)> {
        assert!(p.is_prime_bpsw(), "factor_prime: p must be prime");
        if p.divides(&self.index) {
            self.factor_prime_general(p)
        } else {
            self.factor_prime_kummer(p)
        }
    }

    /// Kummer–Dedekind factorization (valid when `p ∤ index`).
    fn factor_prime_kummer(&self, p: &Int) -> Vec<(PrimeIdeal, usize)> {
        use crate::mod_int::ModInt;
        use crate::poly_finite_field::FactorOverField;
        let n = self.n;
        let t = self.field.defining_polynomial();
        // Reduce T mod p.
        let deg = t.degree().expect("defining polynomial is nonzero");
        let coeffs: Vec<ModInt> = (0..=deg)
            .map(|i| {
                let c = t.coeff(i);
                debug_assert!(c.is_integer(), "T must have integer coefficients");
                ModInt::new(c.numerator().clone(), p.clone())
            })
            .collect();
        let tbar = Poly::new(coeffs);
        let factors = tbar.factor();
        let mut out = Vec::new();
        for (gbar, mult) in factors {
            let f = gbar
                .degree()
                .expect("irreducible factor has positive degree");
            // Lift g to a monic integer polynomial, evaluate g(θ) in the power basis.
            let gdeg = f;
            let gcoeffs: Vec<Rational> = (0..=gdeg)
                .map(|i| Rational::from_integer(gbar.coeff(i).to_int()))
                .collect();
            let gtheta = self.field.element(Poly::new(gcoeffs));
            let vc = poly_coords(gtheta.poly(), n);
            let beta: Vec<Int> = self
                .power_to_order(&vc)
                .iter()
                .map(|x| x.to_integer().expect("g(θ) ∈ ℤ[θ] ⊆ O_K"))
                .collect();
            let ideal = self.ideal_p_beta(p, &beta);
            out.push((
                PrimeIdeal {
                    p: p.clone(),
                    f,
                    e: mult,
                    ideal,
                },
                mult,
            ));
        }
        out
    }

    /// General prime factorization, splitting the `𝔽_p`-algebra `O_K/pO_K`
    /// (works for every `p`, including `p | index`).
    fn factor_prime_general(&self, p: &Int) -> Vec<(PrimeIdeal, usize)> {
        let n = self.n;
        // Structure constants of A = O/pO: table[a][b] = ω_a·ω_b (mod p).
        let mut table: Vec<Vec<Vec<Int>>> = alloc::vec![alloc::vec![Vec::new(); n]; n];
        for a in 0..n {
            let mut ea = alloc::vec![Int::ZERO; n];
            ea[a] = Int::ONE;
            for b in 0..n {
                let mut eb = alloc::vec![Int::ZERO; n];
                eb[b] = Int::ONE;
                let prod = self.mul_order_int(&ea, &eb);
                table[a][b] = prod.iter().map(|x| gmod(x, p)).collect();
            }
        }
        let one = self
            .one_coords()
            .iter()
            .map(|x| gmod(x, p))
            .collect::<Vec<_>>();

        // Radical of A (basis of J/pO over GF(p)).
        let jbasis = self.radical_modp(p);
        let (jrref, jpivots) = rref_modp(&jbasis, n, p);
        let r = jpivots.len();
        let dim_b = n - r; // dim of B = A/J
        let free_cols: Vec<usize> = (0..n).filter(|c| !jpivots.contains(c)).collect();

        // reduce mod J: zero out pivot coordinates.
        let reduce_j = |v: &[Int]| -> Vec<Int> {
            let mut w: Vec<Int> = v.iter().map(|x| gmod(x, p)).collect();
            for (row, &c) in jrref.iter().zip(&jpivots) {
                if !w[c].is_zero() {
                    let f = w[c].clone();
                    for i in 0..n {
                        w[i] = gmod(&w[i].sub(&f.mul(&row[i])), p);
                    }
                }
            }
            w
        };
        // Multiplication and power in B.
        let bmul = |x: &[Int], y: &[Int]| reduce_j(&algebra_mul(&table, x, y, n, p));
        let one_b = reduce_j(&one);
        let bpow = |x: &[Int], e: &Int| -> Vec<Int> {
            let mut result = one_b.clone();
            let mut base = x.to_vec();
            for i in 0..e.bit_len() {
                if e.bit(i) {
                    result = bmul(&result, &base);
                }
                base = bmul(&base, &base);
            }
            result
        };

        // Frobenius on B (as a matrix over the free-coordinate basis of B).
        // basis vector for free column fc is e_{fc} (already J-reduced).
        let mut phi = alloc::vec![alloc::vec![Int::ZERO; dim_b]; dim_b];
        for (col, &fc) in free_cols.iter().enumerate() {
            let mut e = alloc::vec![Int::ZERO; n];
            e[fc] = Int::ONE;
            let w = bpow(&e, p); // e_{fc}^p in B
            for (row, &fr) in free_cols.iter().enumerate() {
                phi[row][col] = w[fr].clone();
            }
        }
        // Fixed space F = ker(Phi - I) over GF(p): number of primes g = dim F.
        let mut phi_minus_i = phi.clone();
        for i in 0..dim_b {
            phi_minus_i[i][i] = gmod(&phi_minus_i[i][i].sub(&Int::ONE), p);
        }
        let fixed = kernel_modp(&phi_minus_i, dim_b, p); // vectors in B-coords
        let g = fixed.len();

        // Helper: embed a B-coordinate vector (length dim_b) into a length-n
        // canonical representative (nonzero only on free columns).
        let embed = |bc: &[Int]| -> Vec<Int> {
            let mut v = alloc::vec![Int::ZERO; n];
            for (idx, &fc) in free_cols.iter().enumerate() {
                v[fc] = bc[idx].clone();
            }
            v
        };

        // Lift a GF(p) vector (order coords) to an integer generator in [0,p).
        let lift = |v: &[Int]| -> Vec<Int> { v.iter().map(|x| gmod(x, p)).collect() };

        // p·O generators (for norm / valuation reference) and the radical J as an
        // ideal (used when g == 1).
        let mut j_gens: Vec<Vec<Int>> = jbasis.iter().map(|v| lift(v)).collect();
        for i in 0..n {
            let mut e = alloc::vec![Int::ZERO; n];
            e[i] = p.clone();
            j_gens.push(e);
        }

        let p_ideal = {
            let mut gens = Vec::new();
            for i in 0..n {
                let mut e = alloc::vec![Int::ZERO; n];
                e[i] = p.clone();
                gens.push(e);
            }
            // pO as ℤ-lattice: rows are p·ω_k.
            Ideal {
                order: self.clone(),
                n,
                basis: hnf_top_n(&gens, n),
            }
        };

        let mut result: Vec<(PrimeIdeal, usize)> = Vec::new();

        if g <= 1 {
            // Single prime above p: 𝔭 = J (the radical), f = dim B, e = n/f.
            let ideal = {
                Ideal {
                    order: self.clone(),
                    n,
                    basis: hnf_top_n(&j_gens, n),
                }
            };
            let f = dim_b.max(1);
            let e = ramification(&ideal, &p_ideal);
            result.push((
                PrimeIdeal {
                    p: p.clone(),
                    f,
                    e,
                    ideal,
                },
                e,
            ));
            return result;
        }

        // Find a separating element b ∈ F: b^p = b and its minimal polynomial in
        // B has degree g (distinct values on all components).
        let sep = self.find_separating(&fixed, &bmul, &one_b, g, p, &embed);
        let b = sep;

        // Minimal polynomial of b over GF(p): powers 1, b, b², … until dependent.
        let (minpoly, _powers) = min_poly_in_algebra(&b, &bmul, &one_b, n, p);
        // Roots of minpoly in GF(p) (it splits into distinct linear factors).
        let roots = poly_roots_modp(&minpoly, p);

        for (i, lam_i) in roots.iter().enumerate() {
            // Idempotent E_i = ∏_{j≠i} (b - λ_j) / (λ_i - λ_j) in B.
            let mut e_idem = one_b.clone();
            for (j, lam_j) in roots.iter().enumerate() {
                if i == j {
                    continue;
                }
                // factor = (b - λ_j·1) · (λ_i - λ_j)⁻¹
                let diff = gmod(&lam_i.sub(lam_j), p);
                let dinv = diff.modinv(p).expect("distinct roots ⇒ invertible");
                let shifted = vec_sub_modp(&b, &vec_scale_modp(lam_j, &one_b, p), p);
                let scaled = vec_scale_modp(&dinv, &shifted, p);
                e_idem = bmul(&e_idem, &scaled);
            }
            // f_i = dim of E_i·B.
            let mut comp_rows: Vec<Vec<Int>> = Vec::new();
            for &fc in &free_cols {
                let mut e = alloc::vec![Int::ZERO; n];
                e[fc] = Int::ONE;
                comp_rows.push(bmul(&e_idem, &e));
            }
            let f_i = rank_modp(&comp_rows, n, p);
            // 𝔭_i = J + (1 - E_i)·B  (as GF(p)-subspace of A); lift to a lattice.
            let one_minus = vec_sub_modp(&one_b, &e_idem, p);
            let mut gens: Vec<Vec<Int>> = j_gens.clone();
            for &fc in &free_cols {
                let mut e = alloc::vec![Int::ZERO; n];
                e[fc] = Int::ONE;
                gens.push(lift(&bmul(&one_minus, &e)));
            }
            let ideal = Ideal {
                order: self.clone(),
                n,
                basis: hnf_top_n(&gens, n),
            };
            let e_ram = ramification(&ideal, &p_ideal);
            result.push((
                PrimeIdeal {
                    p: p.clone(),
                    f: f_i,
                    e: e_ram,
                    ideal,
                },
                e_ram,
            ));
        }
        result
    }

    /// Searches `F` (basis of the Frobenius-fixed subspace, in B-coordinates) for
    /// an element whose minimal polynomial in `B` has degree `g` (separates all
    /// components).
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn find_separating(
        &self,
        fixed: &[Vec<Int>],
        bmul: &dyn Fn(&[Int], &[Int]) -> Vec<Int>,
        one_b: &[Int],
        g: usize,
        p: &Int,
        embed: &dyn Fn(&[Int]) -> Vec<Int>,
    ) -> Vec<Int> {
        let n = self.n;
        // Candidate elements: each fixed basis vector, then random combinations.
        let mut candidates: Vec<Vec<Int>> = Vec::new();
        for fb in fixed {
            candidates.push(embed(fb));
        }
        let mut rng = SeedRng::new(0x1234_5678_9abc_def1 | 1);
        for _ in 0..256 {
            let mut combo = alloc::vec![Int::ZERO; fixed.len()];
            for c in combo.iter_mut() {
                *c = Int::random_below(p, &mut rng).unwrap_or(Int::ZERO);
            }
            let mut v = alloc::vec![Int::ZERO; n];
            for (coef, fb) in combo.iter().zip(fixed) {
                let e = embed(fb);
                for i in 0..n {
                    v[i] = gmod(&v[i].add(&coef.mul(&e[i])), p);
                }
            }
            candidates.push(v);
        }
        for cand in candidates {
            let (mp, _) = min_poly_in_algebra(&cand, bmul, one_b, n, p);
            // degree of minpoly = len - 1
            if mp.len() == g + 1 {
                return cand;
            }
        }
        panic!(
            "factor_prime: no separating element found (p may be smaller than the number of primes above it)"
        );
    }
}

/// The ramification index `e = v_𝔭(pO)`: the largest `m` with `pO ⊆ 𝔭^m`.
fn ramification(prime: &Ideal, p_ideal: &Ideal) -> usize {
    let mut m = 1usize;
    let mut power = prime.clone();
    loop {
        // power == 𝔭^m; check 𝔭^{m+1} ⊇ pO.
        let next = power.mul(prime);
        if next.contains(p_ideal) {
            power = next;
            m += 1;
            if m > 64 {
                break; // safety
            }
        } else {
            break;
        }
    }
    m
}

/// The minimal polynomial of `x` in a commutative `GF(p)`-algebra (given by
/// `mul` and unit `one`), returned as coefficients low-to-high (monic), together
/// with the list of computed powers `1, x, x², …`.
#[allow(clippy::type_complexity)]
fn min_poly_in_algebra(
    x: &[Int],
    mul: &dyn Fn(&[Int], &[Int]) -> Vec<Int>,
    one: &[Int],
    n: usize,
    p: &Int,
) -> (Vec<Int>, Vec<Vec<Int>>) {
    // Build powers until they become linearly dependent over GF(p).
    let mut powers: Vec<Vec<Int>> = alloc::vec![one.to_vec()];
    loop {
        let next = mul(powers.last().unwrap(), x);
        // Check whether `next` is in the span of the existing powers.
        // Solve span(powers)·c = next; if solvable, we have the min poly.
        if let Some(coeffs) = solve_span_modp(&powers, &next, n, p) {
            // next = Σ coeffs[i]·powers[i] ⇒ minpoly = x^d - Σ coeffs[i] x^i.
            let d = powers.len();
            let mut mp = alloc::vec![Int::ZERO; d + 1];
            mp[d] = Int::ONE;
            for (i, c) in coeffs.iter().enumerate() {
                mp[i] = gmod(&c.neg(), p);
            }
            return (mp, powers);
        }
        powers.push(next);
        if powers.len() > n {
            // Should not happen; guard.
            let d = powers.len() - 1;
            let mut mp = alloc::vec![Int::ZERO; d + 1];
            mp[d] = Int::ONE;
            return (mp, powers);
        }
    }
}

/// Solves `Σ cᵢ · vecs[i] = target` over `GF(p)`, returning the coefficients if
/// the target lies in the span (the `vecs` are assumed independent).
fn solve_span_modp(vecs: &[Vec<Int>], target: &[Int], n: usize, p: &Int) -> Option<Vec<Int>> {
    let k = vecs.len();
    // Augmented matrix: columns are vecs (as columns) plus target; solve.
    // Build k×(?) system: we have n equations, k unknowns.
    // Represent as rows = equations (n), cols = k unknowns; RHS = target.
    let mut aug: Vec<Vec<Int>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row: Vec<Int> = (0..k).map(|j| gmod(&vecs[j][i], p)).collect();
        row.push(gmod(&target[i], p));
        aug.push(row);
    }
    // Gaussian elimination on the (k+1)-wide augmented system.
    let width = k + 1;
    let mut pivot_row = 0usize;
    let mut where_piv = alloc::vec![usize::MAX; k];
    for col in 0..k {
        let piv = (pivot_row..n).find(|&i| !aug[i][col].is_zero());
        let piv = match piv {
            Some(pv) => pv,
            None => continue,
        };
        aug.swap(pivot_row, piv);
        let inv = aug[pivot_row][col].modinv(p).expect("prime modulus");
        for c in 0..width {
            aug[pivot_row][c] = gmod(&aug[pivot_row][c].mul(&inv), p);
        }
        for i in 0..n {
            if i == pivot_row || aug[i][col].is_zero() {
                continue;
            }
            let f = aug[i][col].clone();
            for c in 0..width {
                aug[i][c] = gmod(&aug[i][c].sub(&f.mul(&aug[pivot_row][c])), p);
            }
        }
        where_piv[col] = pivot_row;
        pivot_row += 1;
    }
    // Check consistency: any row with all-zero coefficients but nonzero RHS ⇒ no solution.
    for row in &aug {
        if (0..k).all(|c| row[c].is_zero()) && !row[k].is_zero() {
            return None;
        }
    }
    let mut sol = alloc::vec![Int::ZERO; k];
    for (col, &pr) in where_piv.iter().enumerate() {
        if pr != usize::MAX {
            sol[col] = aug[pr][k].clone();
        }
    }
    Some(sol)
}

/// The roots in `GF(p)` of a polynomial given by coefficients low-to-high.
fn poly_roots_modp(coeffs: &[Int], p: &Int) -> Vec<Int> {
    let pe = p.to_i64().expect("prime fits in i64 for root search");
    let mut roots = Vec::new();
    for x in 0..pe {
        let xv = Int::from_i64(x);
        // Horner evaluation mod p.
        let mut acc = Int::ZERO;
        for c in coeffs.iter().rev() {
            acc = gmod(&acc.mul(&xv).add(c), p);
        }
        if acc.is_zero() {
            roots.push(xv);
        }
    }
    roots
}

/// `a · b` for two rational matrices (helper to avoid inherent/trait ambiguity).
fn mat_mul_rat(a: &Matrix<Rational>, b: &Matrix<Rational>) -> Matrix<Rational> {
    a.mul(b)
}

// ===========================================================================
// Ideal.
// ===========================================================================

/// An ideal of the ring of integers `O_K`, stored by the Hermite-normal-form
/// ℤ-basis of its elements: an `n × n` integer matrix whose **rows** are the
/// basis vectors, expressed in the integral basis of `O_K`.
#[derive(Clone)]
pub struct Ideal {
    order: Order,
    n: usize,
    /// Rows = ℤ-basis (order coordinates), in canonical row Hermite normal form.
    basis: Matrix<Int>,
}

impl fmt::Debug for Ideal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ideal(norm {})", self.norm())
    }
}

impl Ideal {
    /// The order this ideal belongs to.
    pub fn order(&self) -> Order {
        self.order.clone()
    }

    /// The HNF ℤ-basis matrix (rows = basis vectors in order coordinates).
    pub fn basis(&self) -> &Matrix<Int> {
        &self.basis
    }

    /// The absolute norm `N(𝔞) = [O_K : 𝔞] = |det(basis)|`.
    pub fn norm(&self) -> Int {
        self.basis.determinant().abs()
    }

    /// The `i`-th basis vector (order coordinates).
    fn row(&self, i: usize) -> Vec<Int> {
        (0..self.n).map(|k| self.basis.get(i, k).clone()).collect()
    }

    /// The product ideal `self · other`: the ℤ-span of all `n²` products of the
    /// two bases, reduced back to `n` generators by Hermite normal form.
    pub fn mul(&self, other: &Ideal) -> Ideal {
        let n = self.n;
        let mut rows: Vec<Vec<Int>> = Vec::with_capacity(n * n);
        for i in 0..n {
            let ai = self.row(i);
            for j in 0..n {
                let bj = other.row(j);
                rows.push(self.order.mul_order_int(&ai, &bj));
            }
        }
        Ideal {
            order: self.order.clone(),
            n,
            basis: hnf_top_n(&rows, n),
        }
    }

    /// The sum ideal `self + other` (the ℤ-span of the union of bases).
    pub fn add(&self, other: &Ideal) -> Ideal {
        let n = self.n;
        let mut rows: Vec<Vec<Int>> = Vec::with_capacity(2 * n);
        for i in 0..n {
            rows.push(self.row(i));
        }
        for i in 0..n {
            rows.push(other.row(i));
        }
        Ideal {
            order: self.order.clone(),
            n,
            basis: hnf_top_n(&rows, n),
        }
    }

    /// `self` raised to the power `k` (`k = 0` giving the unit ideal `O_K`).
    pub fn pow(&self, k: usize) -> Ideal {
        let mut result = self.order.unit_ideal();
        for _ in 0..k {
            result = result.mul(self);
        }
        result
    }

    /// Whether `self ⊇ other` (i.e. `other ⊆ self`).
    pub fn contains(&self, other: &Ideal) -> bool {
        // self ⊇ other ⟺ self + other == self.
        self.add(other).basis == self.basis
    }

    /// Whether the two ideals are equal.
    pub fn equals(&self, other: &Ideal) -> bool {
        self.basis == other.basis
    }

    /// Whether this is the unit ideal `O_K`.
    pub fn is_unit(&self) -> bool {
        self.basis == Matrix::<Int>::identity(self.n)
    }
}

// ===========================================================================
// PrimeIdeal.
// ===========================================================================

/// A prime ideal `𝔭` of `O_K` lying above a rational prime `p`, carrying its
/// residue degree `f` (so `O_K/𝔭 ≅ 𝔽_{p^f}`) and ramification index `e`.
#[derive(Clone)]
pub struct PrimeIdeal {
    p: Int,
    f: usize,
    e: usize,
    ideal: Ideal,
}

impl fmt::Debug for PrimeIdeal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PrimeIdeal(p={}, f={}, e={})", self.p, self.f, self.e)
    }
}

impl PrimeIdeal {
    /// The rational prime `p` below `𝔭`.
    pub fn rational_prime(&self) -> Int {
        self.p.clone()
    }

    /// The residue degree `f` (`N(𝔭) = p^f`).
    #[inline]
    pub fn residue_degree(&self) -> usize {
        self.f
    }

    /// The ramification index `e`.
    #[inline]
    pub fn ramification(&self) -> usize {
        self.e
    }

    /// The underlying ideal.
    pub fn ideal(&self) -> &Ideal {
        &self.ideal
    }

    /// The norm `N(𝔭) = p^f`.
    pub fn norm(&self) -> Int {
        self.p.pow(self.f as u32)
    }
}

// ===========================================================================
// NumberField::maximal_order.
// ===========================================================================

impl NumberField {
    /// Computes the **maximal order** `O_K` (ring of integers) via the Round 2 /
    /// Pohst–Zassenhaus algorithm (CCANT Algorithm 6.1.8).
    ///
    /// Requires the defining polynomial `T` to be **monic with integer
    /// coefficients** (so that `θ` is an algebraic integer and `ℤ[θ] ⊆ O_K`);
    /// this holds for every field built from an integer polynomial.
    ///
    /// # Panics
    /// If `T` has a non-integer coefficient.
    pub fn maximal_order(&self) -> Order {
        let n = self.degree();
        let t = self.defining_polynomial();
        assert!(
            (0..=n).all(|i| t.coeff(i).is_integer()),
            "maximal_order: defining polynomial must have integer coefficients"
        );
        let disc_t = self
            .discriminant()
            .to_integer()
            .expect("disc(T) is an integer for a monic integer polynomial");
        // Start from ℤ[θ].
        let ident = Matrix::<Rational>::identity(n);
        let mut order = Order {
            field: self.clone(),
            n,
            basis: ident.clone(),
            basis_inv: ident,
            disc_t: disc_t.clone(),
            index: Int::ONE,
            d_k: disc_t.clone(),
        };
        // Make p-maximal for each prime with p² | disc(T).
        for (p, e) in disc_t.factor_exponents() {
            if e >= 2 {
                order = order.make_p_maximal(&p);
            }
        }
        order.finalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn q(v: i64) -> Rational {
        Rational::from_integer(Int::from_i64(v))
    }

    fn poly(coeffs: &[i64]) -> Poly<Rational> {
        Poly::new(coeffs.iter().map(|&c| q(c)).collect())
    }

    fn field(coeffs: &[i64]) -> NumberField {
        NumberField::new(poly(coeffs)).unwrap()
    }

    fn ii(v: i64) -> Int {
        Int::from_i64(v)
    }

    // ---- maximal order / discriminant ----

    #[test]
    fn max_order_qsqrt_neg5() {
        // ℚ(√−5): x²+5, d_K = −20, O_K = ℤ[√−5], index 1.
        let k = field(&[5, 0, 1]);
        let o = k.maximal_order();
        assert_eq!(o.index(), ii(1));
        assert_eq!(o.discriminant(), ii(-20));
        // disc(T) = index² · d_K.
        assert_eq!(o.discriminant().mul(&o.index().square()), ii(-20));
    }

    #[test]
    fn max_order_qsqrt5_index2() {
        // ℚ(√5): x²−5, disc(T)=20, d_K=5, index 2 (O_K = ℤ[(1+√5)/2]).
        let k = field(&[-5, 0, 1]);
        let o = k.maximal_order();
        assert_eq!(o.index(), ii(2));
        assert_eq!(o.discriminant(), ii(5));
        // The integral basis must include a half-integer entry (the golden basis).
        let b = o.integral_basis();
        let mut has_half = false;
        for x in b.as_slice() {
            if !x.is_integer() {
                has_half = true;
            }
        }
        assert!(has_half, "ℚ(√5) integral basis should not be ℤ[θ]");
        // disc(T) = index²·d_K.
        assert_eq!(o.discriminant().mul(&o.index().square()), ii(20));
    }

    #[test]
    fn max_order_cbrt2() {
        // ℚ(∛2): x³−2, d_K = −108, O_K = ℤ[∛2], index 1.
        let k = field(&[-2, 0, 0, 1]);
        let o = k.maximal_order();
        assert_eq!(o.index(), ii(1));
        assert_eq!(o.discriminant(), ii(-108));
    }

    #[test]
    fn max_order_cyclo5() {
        // 5th cyclotomic x⁴+x³+x²+x+1: d_K = 125, index 1.
        let k = field(&[1, 1, 1, 1, 1]);
        let o = k.maximal_order();
        assert_eq!(o.index(), ii(1));
        assert_eq!(o.discriminant(), ii(125));
    }

    #[test]
    fn order_contains_one_and_theta_closed_under_mul() {
        for coeffs in [
            vec![5i64, 0, 1],
            vec![-5, 0, 1],
            vec![-2, 0, 0, 1],
            vec![1, 1, 1, 1, 1],
        ] {
            let k = field(&coeffs);
            let o = k.maximal_order();
            let n = o.degree();
            // 1 and θ have integer order coordinates.
            let one = o.one_coords();
            assert_eq!(one.len(), n);
            let mut theta_power = vec![Rational::ZERO; n];
            theta_power[1] = Rational::ONE;
            let theta = o.power_to_order(&theta_power);
            assert!(theta.iter().all(|x| x.is_integer()), "θ ∈ O_K");
            // Closure: ω_i·ω_j has integer order coordinates.
            for i in 0..n {
                for j in 0..n {
                    let mut ei = vec![Int::ZERO; n];
                    ei[i] = Int::ONE;
                    let mut ej = vec![Int::ZERO; n];
                    ej[j] = Int::ONE;
                    let _ = o.mul_order_int(&ei, &ej); // panics if non-integral
                }
            }
        }
    }

    // ---- prime factorization (Kummer–Dedekind, index 1) ----

    /// Checks the invariants for the factorization of `p` in `O`.
    fn check_factorization(o: &Order, p: &Int) -> Vec<(PrimeIdeal, usize)> {
        let n = o.degree();
        let facs = o.factor_prime(p);
        // Σ eᵢ·fᵢ = n.
        let sum: usize = facs.iter().map(|(pr, e)| e * pr.residue_degree()).sum();
        assert_eq!(sum, n, "Σ eᵢfᵢ = n for p={p}");
        // ∏ N(𝔭ᵢ)^{eᵢ} = pⁿ.
        let mut prod = Int::ONE;
        for (pr, e) in &facs {
            prod = prod.mul(&pr.norm().pow(*e as u32));
        }
        assert_eq!(prod, p.pow(n as u32), "∏ N(𝔭)^e = pⁿ for p={p}");
        // Each 𝔭ᵢ has norm p^{fᵢ}.
        for (pr, _) in &facs {
            assert_eq!(pr.ideal().norm(), pr.norm());
        }
        // ∏ 𝔭ᵢ^{eᵢ} = pO_K.
        let mut acc = o.unit_ideal();
        for (pr, e) in &facs {
            acc = acc.mul(&pr.ideal().pow(*e));
        }
        // pO_K.
        let mut p_power = vec![Rational::ZERO; n];
        p_power[0] = Rational::from_integer(p.clone());
        let po = o.principal_ideal(&p_power);
        assert!(acc.equals(&po), "∏ 𝔭^e = pO_K for p={p}");
        facs
    }

    #[test]
    fn factor_qsqrt_neg5() {
        let k = field(&[5, 0, 1]); // x²+5
        let o = k.maximal_order();
        // 2 ramified: 2 = 𝔭².
        let f2 = check_factorization(&o, &ii(2));
        assert_eq!(f2.len(), 1);
        assert_eq!(f2[0].1, 2); // e = 2
        assert_eq!(f2[0].0.residue_degree(), 1);
        // 3 split: 3 = 𝔭𝔮.
        let f3 = check_factorization(&o, &ii(3));
        assert_eq!(f3.len(), 2);
        // 5 ramified.
        let f5 = check_factorization(&o, &ii(5));
        assert_eq!(f5.len(), 1);
        assert_eq!(f5[0].1, 2);
        // 7 split (−5 is a QR mod 7: 3²=2? -5≡2, and 2 is not a QR mod 7 → inert).
        let f7 = check_factorization(&o, &ii(7));
        // −5 mod 7 = 2; QRs mod 7 are {1,2,4}; 2 is a QR ⇒ split.
        assert_eq!(f7.len(), 2);
        // 11: −5 mod 11 = 6; QRs mod 11 {1,3,4,5,9}; 6 not QR ⇒ inert.
        let f11 = check_factorization(&o, &ii(11));
        assert_eq!(f11.len(), 1);
        assert_eq!(f11[0].0.residue_degree(), 2);
        // 23: −5 mod 23 = 18; check split/inert via invariants only.
        let _ = check_factorization(&o, &ii(23));
    }

    #[test]
    fn factor_cbrt2() {
        let k = field(&[-2, 0, 0, 1]); // x³−2
        let o = k.maximal_order();
        // 2 totally ramified: 2 = 𝔭³.
        let f2 = check_factorization(&o, &ii(2));
        assert_eq!(f2.len(), 1);
        assert_eq!(f2[0].1, 3);
        // 3 totally ramified (3 | disc = −108).
        let f3 = check_factorization(&o, &ii(3));
        assert_eq!(f3.len(), 1);
        assert_eq!(f3[0].1, 3);
        // 5: 3 is a cube root of 2 mod 5, so x³−2 ≡ (x−3)(x²+3x+4); 5 = 𝔭₁𝔭₂
        // with residue degrees 1 and 2.
        let f5 = check_factorization(&o, &ii(5));
        assert_eq!(f5.len(), 2);
        let mut degs: Vec<usize> = f5.iter().map(|(pr, _)| pr.residue_degree()).collect();
        degs.sort_unstable();
        assert_eq!(degs, vec![1, 2]);
    }

    #[test]
    fn factor_cyclo5() {
        let k = field(&[1, 1, 1, 1, 1]); // x⁴+x³+x²+x+1
        let o = k.maximal_order();
        // 5 totally ramified (5 | d_K = 125): 5 = 𝔭⁴.
        let f5 = check_factorization(&o, &ii(5));
        assert_eq!(f5.len(), 1);
        assert_eq!(f5[0].1, 4);
        // 11 ≡ 1 mod 5 ⇒ splits completely into 4 primes of degree 1.
        let f11 = check_factorization(&o, &ii(11));
        assert_eq!(f11.len(), 4);
        // 2 has order 4 mod 5 ⇒ inert (single prime of degree 4).
        let f2 = check_factorization(&o, &ii(2));
        assert_eq!(f2.len(), 1);
        assert_eq!(f2[0].0.residue_degree(), 4);
        // 3 has order 4 mod 5 ⇒ inert.
        let f3 = check_factorization(&o, &ii(3));
        assert_eq!(f3.len(), 1);
        assert_eq!(f3[0].0.residue_degree(), 4);
    }

    // ---- ramified primes are exactly those dividing d_K ----

    #[test]
    fn ramified_iff_divides_dk() {
        let k = field(&[5, 0, 1]); // ℚ(√−5), d_K = −20 = −2²·5
        let o = k.maximal_order();
        for p in [2i64, 3, 5, 7, 11, 13] {
            let pi = ii(p);
            let facs = o.factor_prime(&pi);
            let ramified = facs.iter().any(|(pr, _)| pr.ramification() >= 2);
            let divides = pi.divides(&o.discriminant());
            assert_eq!(ramified, divides, "ramification vs d_K for p={p}");
        }
    }

    // ---- p | index general splitting ----

    #[test]
    fn factor_qsqrt5_p2_index2() {
        // ℚ(√5): index 2, so factoring p=2 hits p | index (general path).
        // −5 ≡ 5 mod 8 ⇒ 2 is inert in ℚ(√5): 2 = 𝔭, f = 2, e = 1.
        let k = field(&[-5, 0, 1]);
        let o = k.maximal_order();
        assert_eq!(o.index(), ii(2));
        let facs = check_factorization(&o, &ii(2));
        assert_eq!(facs.len(), 1);
        assert_eq!(facs[0].0.residue_degree(), 2);
        assert_eq!(facs[0].1, 1);
    }

    #[test]
    fn factor_qsqrt_neg7_p2_split_index2() {
        // ℚ(√−7): x²+7, disc=−28, d_K=−7, index 2. −7 ≡ 1 mod 8 ⇒ 2 splits:
        // 2 = 𝔭𝔮 with p | index (general path).
        let k = field(&[7, 0, 1]);
        let o = k.maximal_order();
        assert_eq!(o.index(), ii(2));
        assert_eq!(o.discriminant(), ii(-7));
        let facs = check_factorization(&o, &ii(2));
        assert_eq!(facs.len(), 2, "2 splits in ℚ(√−7)");
        for (pr, e) in &facs {
            assert_eq!(pr.residue_degree(), 1);
            assert_eq!(*e, 1);
        }
        // 7 is ramified (7 | d_K) and 7 ∤ index ⇒ Kummer path.
        let f7 = check_factorization(&o, &ii(7));
        assert_eq!(f7.len(), 1);
        assert_eq!(f7[0].1, 2);
    }

    // ---- ideal arithmetic sanity ----

    #[test]
    fn ideal_norm_and_unit() {
        let k = field(&[5, 0, 1]); // ℚ(√−5)
        let o = k.maximal_order();
        let unit = o.unit_ideal();
        assert!(unit.is_unit());
        assert_eq!(unit.norm(), ii(1));
        // (√−5) has norm |N(√−5)| = 5.
        let sqrt = o.principal_ideal(&[q(0), q(1)]); // θ = √−5
        assert_eq!(sqrt.norm(), ii(5));
        // (2) has norm 4.
        let two = o.principal_ideal(&[q(2), q(0)]);
        assert_eq!(two.norm(), ii(4));
        // multiplicativity of norm on principal ideals: (2)·(√−5) = (2√−5), norm 20.
        let prod = two.mul(&sqrt);
        assert_eq!(prod.norm(), ii(20));
    }
}
