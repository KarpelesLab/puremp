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
use crate::nat::{Nat, Reciprocal};

/// The shared modulus context (modulus + precomputed reciprocal).
struct Ring {
    modulus: Nat,
    recip: Reciprocal,
}

/// An element of `ℤ/mℤ`, represented by its canonical residue in `[0, m)`.
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
        let ring = Rc::new(Ring {
            recip: Reciprocal::new(&m),
            modulus: m,
        });
        let residue = value.rem_euclid(&modulus).magnitude();
        ModInt {
            value: residue,
            ring,
        }
    }

    /// Builds another residue in the *same* ring as `self` (sharing the modulus
    /// and its precomputed reciprocal).
    pub fn of(&self, value: Int) -> ModInt {
        let residue = value
            .rem_euclid(&Int::from(self.ring.modulus.clone()))
            .magnitude();
        ModInt {
            value: residue,
            ring: self.ring.clone(),
        }
    }

    /// Returns the canonical residue in `[0, m)`.
    #[inline]
    pub fn residue(&self) -> &Nat {
        &self.value
    }

    /// Returns the residue as an [`Int`] in `[0, m)`.
    #[inline]
    pub fn to_int(&self) -> Int {
        Int::from(self.value.clone())
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
        self.wrap(self.ring.recip.reduce(&self.value.mul(&rhs.value)))
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
        let inv = self.to_int().modinv(&self.modulus())?;
        Some(self.wrap(inv.magnitude()))
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
        self.wrap(self.value.modpow(&exp.magnitude(), &self.ring.modulus))
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
        fmt::Display::fmt(&self.value, f)
    }
}

impl fmt::Debug for ModInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ModInt({} mod {})", self.value, self.ring.modulus)
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
