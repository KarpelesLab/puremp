//! Modular integers — residue classes `ℤ/mℤ` for a fixed modulus.
//!
//! [`ModInt`] carries its modulus (and a precomputed Barrett [`Reciprocal`]) so
//! `+ - * /` and `pow` reduce automatically, giving natural-looking modular
//! arithmetic. The modulus is shared cheaply (reference-counted) between values
//! produced from one another, so building a family of residues via [`ModInt::of`]
//! avoids recomputing the reciprocal.

use core::cmp::Ordering;
use core::fmt;

use alloc::rc::Rc;

use crate::int::Int;
use crate::nat::{MontCtx, Nat, Reciprocal};

/// How a ring reduces products: Montgomery for odd moduli (values are held in
/// Montgomery form `x·R mod m`), Barrett for even ones (values are the plain
/// canonical residue `x mod m`).
enum Backend {
    /// Even modulus: no Montgomery. Stored value is the canonical residue.
    Barrett(Reciprocal),
    /// Odd modulus: stored value is the Montgomery representative `x·R mod m`.
    Mont(MontCtx),
}

/// The shared modulus context (modulus + the reduction backend).
struct Ring {
    modulus: Nat,
    backend: Backend,
}

impl Ring {
    /// Builds the shared ring for modulus `m >= 2`, choosing Montgomery for odd
    /// `m` and Barrett for even `m`.
    fn new(m: Nat) -> Ring {
        let backend = if m.is_even() {
            Backend::Barrett(Reciprocal::new(&m))
        } else {
            Backend::Mont(MontCtx::new(&m))
        };
        Ring {
            modulus: m,
            backend,
        }
    }

    /// Maps a canonical residue in `[0, m)` to the ring's internal stored form
    /// (Montgomery representative for odd `m`, identity for even `m`).
    fn encode(&self, residue: Nat) -> Nat {
        match &self.backend {
            Backend::Barrett(_) => residue,
            Backend::Mont(ctx) => ctx.to_mont(&residue),
        }
    }

    /// Maps an internal stored value back to its canonical residue in `[0, m)`.
    fn decode(&self, value: &Nat) -> Nat {
        match &self.backend {
            Backend::Barrett(_) => value.clone(),
            Backend::Mont(ctx) => ctx.to_residue(value),
        }
    }
}

/// An element of `ℤ/mℤ`. Its canonical residue lives in `[0, m)`; internally the
/// value is held in Montgomery form for odd moduli (so `*`/`pow` skip the
/// separate reduction) and as the plain residue for even moduli.
#[derive(Clone)]
pub struct ModInt {
    value: Nat,
    ring: Rc<Ring>,
}

impl ModInt {
    /// Builds `value mod modulus`. Panics if `modulus < 2`.
    pub fn new(value: Int, modulus: Int) -> ModInt {
        assert!(modulus > Int::ONE, "ModInt: modulus must be >= 2");
        let m = modulus.magnitude();
        let ring = Rc::new(Ring::new(m));
        let residue = value.rem_euclid(&modulus).magnitude();
        ModInt {
            value: ring.encode(residue),
            ring,
        }
    }

    /// Builds another residue in the *same* ring as `self` (sharing the modulus
    /// and its precomputed reduction backend).
    pub fn of(&self, value: Int) -> ModInt {
        let residue = value
            .rem_euclid(&Int::from(self.ring.modulus.clone()))
            .magnitude();
        ModInt {
            value: self.ring.encode(residue),
            ring: self.ring.clone(),
        }
    }

    /// Returns the canonical residue in `[0, m)`.
    #[inline]
    pub fn residue(&self) -> Nat {
        self.ring.decode(&self.value)
    }

    /// Returns the residue as an [`Int`] in `[0, m)`.
    #[inline]
    pub fn to_int(&self) -> Int {
        Int::from(self.ring.decode(&self.value))
    }

    /// Returns the modulus.
    #[inline]
    pub fn modulus(&self) -> Int {
        Int::from(self.ring.modulus.clone())
    }

    /// Returns `true` if this residue is zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.value.is_zero()
    }

    fn same_ring(&self, other: &ModInt) {
        debug_assert!(
            Rc::ptr_eq(&self.ring, &other.ring) || self.ring.modulus == other.ring.modulus,
            "ModInt: operands have different moduli"
        );
    }

    fn wrap(&self, value: Nat) -> ModInt {
        ModInt {
            value,
            ring: self.ring.clone(),
        }
    }

    /// Returns `self + rhs`.
    pub fn add(&self, rhs: &ModInt) -> ModInt {
        self.same_ring(rhs);
        let s = self.value.add(&rhs.value);
        let m = &self.ring.modulus;
        let v = if s.cmp(m) != Ordering::Less {
            s.checked_sub(m).unwrap()
        } else {
            s
        };
        self.wrap(v)
    }

    /// Returns `self - rhs`.
    pub fn sub(&self, rhs: &ModInt) -> ModInt {
        self.same_ring(rhs);
        let m = &self.ring.modulus;
        let v = if self.value.cmp(&rhs.value) != Ordering::Less {
            self.value.checked_sub(&rhs.value).unwrap()
        } else {
            self.value.add(m).checked_sub(&rhs.value).unwrap()
        };
        self.wrap(v)
    }

    /// Returns `self · rhs`.
    pub fn mul(&self, rhs: &ModInt) -> ModInt {
        self.same_ring(rhs);
        let v = match &self.ring.backend {
            Backend::Barrett(recip) => recip.reduce(&self.value.mul(&rhs.value)),
            Backend::Mont(ctx) => ctx.mul(&self.value, &rhs.value),
        };
        self.wrap(v)
    }

    /// Returns `-self`.
    pub fn neg(&self) -> ModInt {
        if self.value.is_zero() {
            self.clone()
        } else {
            self.wrap(self.ring.modulus.checked_sub(&self.value).unwrap())
        }
    }

    /// Returns the modular inverse `self⁻¹`, or `None` if `gcd(self, m) != 1`.
    pub fn inv(&self) -> Option<ModInt> {
        match &self.ring.backend {
            // Barrett/even: the stored value is the true residue; invert directly.
            Backend::Barrett(_) => {
                let inv = self.to_int().modinv(&self.modulus())?;
                Some(self.wrap(inv.magnitude()))
            }
            // Montgomery/odd: invert in the Montgomery domain (stored form → stored
            // form) without converting out and back in.
            Backend::Mont(ctx) => Some(self.wrap(ctx.inv(&self.value)?)),
        }
    }

    /// Returns `self / rhs = self · rhs⁻¹`. Panics if `rhs` is not invertible.
    pub fn div(&self, rhs: &ModInt) -> ModInt {
        self.mul(&rhs.inv().expect("ModInt::div: divisor is not invertible"))
    }

    /// Returns `self` raised to `exp` (negative exponents use the inverse).
    pub fn pow(&self, exp: &Int) -> ModInt {
        if exp.is_negative() {
            return self
                .inv()
                .expect("ModInt::pow: base not invertible for a negative exponent")
                .pow(&exp.abs());
        }
        let v = match &self.ring.backend {
            Backend::Barrett(_) => self.value.modpow(&exp.magnitude(), &self.ring.modulus),
            Backend::Mont(ctx) => ctx.pow(&self.value, &exp.magnitude()),
        };
        self.wrap(v)
    }
}

impl PartialEq for ModInt {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.ring.modulus == other.ring.modulus
    }
}

impl Eq for ModInt {}

impl fmt::Display for ModInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.residue(), f)
    }
}

impl fmt::Debug for ModInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ModInt({} mod {})", self.residue(), self.ring.modulus)
    }
}

macro_rules! mod_binop {
    ($tr:ident, $m:ident, $atr:ident, $am:ident) => {
        impl core::ops::$tr for ModInt {
            type Output = ModInt;
            #[inline]
            fn $m(self, rhs: ModInt) -> ModInt {
                ModInt::$m(&self, &rhs)
            }
        }
        impl core::ops::$tr<&ModInt> for &ModInt {
            type Output = ModInt;
            #[inline]
            fn $m(self, rhs: &ModInt) -> ModInt {
                ModInt::$m(self, rhs)
            }
        }
        impl core::ops::$atr for ModInt {
            #[inline]
            fn $am(&mut self, rhs: ModInt) {
                *self = ModInt::$m(self, &rhs);
            }
        }
    };
}

mod_binop!(Add, add, AddAssign, add_assign);
mod_binop!(Sub, sub, SubAssign, sub_assign);
mod_binop!(Mul, mul, MulAssign, mul_assign);
mod_binop!(Div, div, DivAssign, div_assign);

impl core::ops::Neg for ModInt {
    type Output = ModInt;
    #[inline]
    fn neg(self) -> ModInt {
        ModInt::neg(&self)
    }
}
