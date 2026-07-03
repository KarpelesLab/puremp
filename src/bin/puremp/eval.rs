//! A tiny tokenizer + recursive-descent evaluator for the REPL, operating over
//! [`puremp::Int`].
//!
//! Grammar (lowest to highest precedence):
//!
//! ```text
//! line    := ident '=' expr | expr
//! expr    := term  (('+' | '-') term)*
//! term    := unary (('*' | '/' | '%') unary)*
//! unary   := ('+' | '-') unary | power
//! power   := primary ('**' unary)?          // right-associative
//! primary := number | ident | '(' expr ')'
//! ```

use std::collections::BTreeMap;

use puremp::Int;

/// Variable environment: name → value.
pub(crate) type Env = BTreeMap<String, Int>;

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Num(Int),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Pow,
    LParen,
    RParen,
    Eq,
}

fn tokenize(input: &str) -> Result<Vec<Tok>, String> {
    let bytes = input.as_bytes();
    let mut toks = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        match c {
            b' ' | b'\t' | b'\r' | b'\n' => i += 1,
            b'+' => {
                toks.push(Tok::Plus);
                i += 1;
            }
            b'-' => {
                toks.push(Tok::Minus);
                i += 1;
            }
            b'*' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'*' {
                    toks.push(Tok::Pow);
                    i += 2;
                } else {
                    toks.push(Tok::Star);
                    i += 1;
                }
            }
            b'/' => {
                toks.push(Tok::Slash);
                i += 1;
            }
            b'%' => {
                toks.push(Tok::Percent);
                i += 1;
            }
            b'(' => {
                toks.push(Tok::LParen);
                i += 1;
            }
            b')' => {
                toks.push(Tok::RParen);
                i += 1;
            }
            b'=' => {
                toks.push(Tok::Eq);
                i += 1;
            }
            b'0'..=b'9' => {
                let start = i;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                let text = &input[start..i];
                let n: Int = text.parse().map_err(|_| format!("bad number `{text}`"))?;
                toks.push(Tok::Num(n));
            }
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

struct Parser<'a> {
    toks: Vec<Tok>,
    pos: usize,
    env: &'a Env,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn bump(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn expr(&mut self) -> Result<Int, String> {
        let mut acc = self.term()?;
        while let Some(op) = self.peek() {
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

    fn term(&mut self) -> Result<Int, String> {
        let mut acc = self.unary()?;
        while let Some(op) = self.peek() {
            match op {
                Tok::Star => {
                    self.pos += 1;
                    acc = acc.mul(&self.unary()?);
                }
                Tok::Slash => {
                    self.pos += 1;
                    let rhs = self.unary()?;
                    acc = acc
                        .div_rem(&rhs)
                        .ok_or_else(|| "division by zero".to_string())?
                        .0;
                }
                Tok::Percent => {
                    self.pos += 1;
                    let rhs = self.unary()?;
                    acc = acc
                        .div_rem(&rhs)
                        .ok_or_else(|| "division by zero".to_string())?
                        .1;
                }
                _ => break,
            }
        }
        Ok(acc)
    }

    fn unary(&mut self) -> Result<Int, String> {
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

    fn power(&mut self) -> Result<Int, String> {
        let base = self.primary()?;
        if matches!(self.peek(), Some(Tok::Pow)) {
            self.pos += 1;
            let exp = self.unary()?;
            let exp_u32 = int_to_exponent(&exp)?;
            Ok(base.pow(exp_u32))
        } else {
            Ok(base)
        }
    }

    fn primary(&mut self) -> Result<Int, String> {
        match self.bump() {
            Some(Tok::Num(n)) => Ok(n),
            Some(Tok::Ident(name)) => self
                .env
                .get(&name)
                .cloned()
                .ok_or_else(|| format!("undefined variable `{name}`")),
            Some(Tok::LParen) => {
                let inner = self.expr()?;
                match self.bump() {
                    Some(Tok::RParen) => Ok(inner),
                    _ => Err("expected `)`".to_string()),
                }
            }
            Some(other) => Err(format!("unexpected token {other:?}")),
            None => Err("unexpected end of input".to_string()),
        }
    }
}

/// Converts an exponent [`Int`] into the `u32` that [`Int::pow`] expects,
/// rejecting negative or oversized exponents with a helpful message.
fn int_to_exponent(exp: &Int) -> Result<u32, String> {
    use puremp::Sign;
    if exp.sign() == Sign::Negative {
        return Err("negative exponent is not an integer".to_string());
    }
    // Round-trip through the decimal string; exponents are tiny in practice.
    exp.to_string()
        .parse::<u32>()
        .map_err(|_| "exponent too large".to_string())
}

/// Evaluates one input line, returning `Some(value)` for an expression or `None`
/// for an assignment.
pub(crate) fn eval_line(line: &str, env: &mut Env) -> Result<Option<Int>, String> {
    let toks = tokenize(line)?;
    if toks.is_empty() {
        return Ok(None);
    }
    // Assignment: `ident = expr`.
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
