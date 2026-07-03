//! Tokenizer + recursive-descent evaluator for the REPL, operating over exact
//! [`puremp::Rational`] values (integers are rationals with denominator one).
//!
//! Grammar (lowest to highest precedence):
//!
//! ```text
//! line    := ident '=' expr | expr
//! expr    := term  (('+' | '-') term)*
//! term    := unary (('*' | '/' | '%') unary)*
//! unary   := ('+' | '-') unary | power
//! power   := primary ('**' unary)?              // right-associative
//! primary := number | ident '(' args ')' | ident | '(' expr ')'
//! number  := decimal | '0x'hex | '0b'bin | '0o'oct
//! ```

use std::collections::BTreeMap;

use puremp::{Int, Rational};

/// Variable environment: name → value.
pub(crate) type Env = BTreeMap<String, Rational>;

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Num(Rational),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Pow,
    LParen,
    RParen,
    Comma,
    Eq,
}

fn parse_number(input: &str, bytes: &[u8], i: &mut usize) -> Result<Tok, String> {
    let start = *i;
    // Non-decimal radix prefixes.
    if bytes[*i] == b'0' && *i + 1 < bytes.len() {
        let (radix, skip) = match bytes[*i + 1] {
            b'x' | b'X' => (16, 2),
            b'b' | b'B' => (2, 2),
            b'o' | b'O' => (8, 2),
            _ => (0, 0),
        };
        if radix != 0 {
            *i += skip;
            let ds = *i;
            while *i < bytes.len() && bytes[*i].is_ascii_alphanumeric() {
                *i += 1;
            }
            let digits = &input[ds..*i];
            let n = Int::from_str_radix(digits, radix)
                .map_err(|_| format!("bad base-{radix} literal `{digits}`"))?;
            return Ok(Tok::Num(Rational::from_integer(n)));
        }
    }
    // Decimal integer or fixed-point (handled by Rational's FromStr).
    while *i < bytes.len() && (bytes[*i].is_ascii_digit() || bytes[*i] == b'.') {
        *i += 1;
    }
    let text = &input[start..*i];
    let r: Rational = text.parse().map_err(|_| format!("bad number `{text}`"))?;
    Ok(Tok::Num(r))
}

fn tokenize(input: &str) -> Result<Vec<Tok>, String> {
    let bytes = input.as_bytes();
    let mut toks = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        match c {
            b' ' | b'\t' | b'\r' | b'\n' => i += 1,
            b'+' => push(&mut toks, Tok::Plus, &mut i),
            b'-' => push(&mut toks, Tok::Minus, &mut i),
            b'*' => {
                if bytes.get(i + 1) == Some(&b'*') {
                    toks.push(Tok::Pow);
                    i += 2;
                } else {
                    push(&mut toks, Tok::Star, &mut i);
                }
            }
            b'/' => push(&mut toks, Tok::Slash, &mut i),
            b'%' => push(&mut toks, Tok::Percent, &mut i),
            b'(' => push(&mut toks, Tok::LParen, &mut i),
            b')' => push(&mut toks, Tok::RParen, &mut i),
            b',' => push(&mut toks, Tok::Comma, &mut i),
            b'=' => push(&mut toks, Tok::Eq, &mut i),
            b'0'..=b'9' | b'.' => toks.push(parse_number(input, bytes, &mut i)?),
            _ if c == b'_' || c.is_ascii_alphabetic() => {
                let start = i;
                while i < bytes.len() && (bytes[i] == b'_' || bytes[i].is_ascii_alphanumeric()) {
                    i += 1;
                }
                toks.push(Tok::Ident(input[start..i].to_string()));
            }
            other => return Err(format!("unexpected character `{}`", other as char)),
        }
    }
    Ok(toks)
}

fn push(toks: &mut Vec<Tok>, t: Tok, i: &mut usize) {
    toks.push(t);
    *i += 1;
}

struct Parser<'a> {
    toks: Vec<Tok>,
    pos: usize,
    env: &'a Env,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn expr(&mut self) -> Result<Rational, String> {
        let mut acc = self.term()?;
        while let Some(op) = self.peek().cloned() {
            match op {
                Tok::Plus => {
                    self.pos += 1;
                    acc = acc.add(&self.term()?);
                }
                Tok::Minus => {
                    self.pos += 1;
                    acc = acc.sub(&self.term()?);
                }
                _ => break,
            }
        }
        Ok(acc)
    }

    fn term(&mut self) -> Result<Rational, String> {
        let mut acc = self.unary()?;
        while let Some(op) = self.peek().cloned() {
            match op {
                Tok::Star => {
                    self.pos += 1;
                    acc = acc.mul(&self.unary()?);
                }
                Tok::Slash => {
                    self.pos += 1;
                    let rhs = self.unary()?;
                    if rhs.is_zero() {
                        return Err("division by zero".to_string());
                    }
                    acc = acc.div(&rhs);
                }
                Tok::Percent => {
                    self.pos += 1;
                    let rhs = self.unary()?;
                    acc = int_rem(&acc, &rhs)?;
                }
                _ => break,
            }
        }
        Ok(acc)
    }

    fn unary(&mut self) -> Result<Rational, String> {
        match self.peek() {
            Some(Tok::Minus) => {
                self.pos += 1;
                Ok(self.unary()?.neg())
            }
            Some(Tok::Plus) => {
                self.pos += 1;
                self.unary()
            }
            _ => self.power(),
        }
    }

    fn power(&mut self) -> Result<Rational, String> {
        let base = self.primary()?;
        if matches!(self.peek(), Some(Tok::Pow)) {
            self.pos += 1;
            let exp = self.unary()?;
            let e = to_exponent(&exp)?;
            Ok(base.pow(e))
        } else {
            Ok(base)
        }
    }

    fn primary(&mut self) -> Result<Rational, String> {
        match self.toks.get(self.pos).cloned() {
            Some(Tok::Num(n)) => {
                self.pos += 1;
                Ok(n)
            }
            Some(Tok::Ident(name)) => {
                self.pos += 1;
                if matches!(self.peek(), Some(Tok::LParen)) {
                    self.pos += 1;
                    let args = self.args()?;
                    call_function(&name, &args)
                } else {
                    self.env
                        .get(&name)
                        .cloned()
                        .ok_or_else(|| format!("undefined variable `{name}`"))
                }
            }
            Some(Tok::LParen) => {
                self.pos += 1;
                let inner = self.expr()?;
                if matches!(self.peek(), Some(Tok::RParen)) {
                    self.pos += 1;
                    Ok(inner)
                } else {
                    Err("expected `)`".to_string())
                }
            }
            Some(other) => Err(format!("unexpected token {other:?}")),
            None => Err("unexpected end of input".to_string()),
        }
    }

    fn args(&mut self) -> Result<Vec<Rational>, String> {
        let mut out = Vec::new();
        if matches!(self.peek(), Some(Tok::RParen)) {
            self.pos += 1;
            return Ok(out);
        }
        loop {
            out.push(self.expr()?);
            match self.peek() {
                Some(Tok::Comma) => self.pos += 1,
                Some(Tok::RParen) => {
                    self.pos += 1;
                    return Ok(out);
                }
                _ => return Err("expected `,` or `)`".to_string()),
            }
        }
    }
}

/// Requires an integer value and returns it as [`Int`].
fn as_int(r: &Rational, what: &str) -> Result<Int, String> {
    r.to_integer()
        .ok_or_else(|| format!("{what} requires an integer, got {r}"))
}

/// Integer (truncated) remainder for the `%` operator.
fn int_rem(a: &Rational, b: &Rational) -> Result<Rational, String> {
    let (ai, bi) = (as_int(a, "`%`")?, as_int(b, "`%`")?);
    let (_, r) = ai
        .div_rem(&bi)
        .ok_or_else(|| "division by zero".to_string())?;
    Ok(Rational::from_integer(r))
}

/// Converts an exponent to `i32` for `Rational::pow`.
fn to_exponent(r: &Rational) -> Result<i32, String> {
    let i = as_int(r, "exponent")?;
    i.to_string()
        .parse::<i32>()
        .map_err(|_| "exponent out of range".to_string())
}

/// Dispatches a built-in function call.
fn call_function(name: &str, args: &[Rational]) -> Result<Rational, String> {
    let arity = |n: usize| -> Result<(), String> {
        if args.len() == n {
            Ok(())
        } else {
            Err(format!(
                "{name}() takes {n} argument(s), got {}",
                args.len()
            ))
        }
    };
    match name {
        "abs" => {
            arity(1)?;
            Ok(args[0].abs())
        }
        "floor" => {
            arity(1)?;
            Ok(Rational::from_integer(args[0].floor()))
        }
        "ceil" => {
            arity(1)?;
            Ok(Rational::from_integer(args[0].ceil()))
        }
        "num" => {
            arity(1)?;
            Ok(Rational::from_integer(args[0].numerator().clone()))
        }
        "den" => {
            arity(1)?;
            Ok(Rational::from_integer(args[0].denominator().clone()))
        }
        "gcd" => {
            arity(2)?;
            Ok(Rational::from_integer(
                as_int(&args[0], "gcd")?.gcd(&as_int(&args[1], "gcd")?),
            ))
        }
        "lcm" => {
            arity(2)?;
            Ok(Rational::from_integer(
                as_int(&args[0], "lcm")?.lcm(&as_int(&args[1], "lcm")?),
            ))
        }
        "isqrt" => {
            arity(1)?;
            let n = as_int(&args[0], "isqrt")?;
            if n.is_negative() {
                return Err("isqrt of a negative number".to_string());
            }
            Ok(Rational::from_integer(Int::from(n.magnitude().isqrt())))
        }
        "fact" => {
            arity(1)?;
            let n = as_int(&args[0], "fact")?;
            let k = n
                .to_string()
                .parse::<u64>()
                .map_err(|_| "fact argument out of range".to_string())?;
            let mut acc = Int::ONE;
            for m in 2..=k {
                acc = acc.mul(&Int::from_i64(m as i64));
            }
            Ok(Rational::from_integer(acc))
        }
        _ => Err(format!("unknown function `{name}`")),
    }
}

/// Evaluates one input line, returning `Some(value)` for an expression or `None`
/// for an assignment.
pub(crate) fn eval_line(line: &str, env: &mut Env) -> Result<Option<Rational>, String> {
    let toks = tokenize(line)?;
    if toks.is_empty() {
        return Ok(None);
    }
    if let (Some(Tok::Ident(name)), Some(Tok::Eq)) = (toks.first(), toks.get(1)) {
        let name = name.clone();
        let mut parser = Parser {
            toks: toks[2..].to_vec(),
            pos: 0,
            env,
        };
        let value = parser.expr()?;
        if parser.pos != parser.toks.len() {
            return Err("trailing tokens after assignment".to_string());
        }
        env.insert(name, value);
        return Ok(None);
    }

    let mut parser = Parser { toks, pos: 0, env };
    let value = parser.expr()?;
    if parser.pos != parser.toks.len() {
        return Err("trailing tokens after expression".to_string());
    }
    Ok(Some(value))
}
