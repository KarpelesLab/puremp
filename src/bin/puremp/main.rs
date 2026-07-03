//! `puremp` — a command-line arbitrary-precision calculator / REPL.
//!
//! Reads expressions on standard input and evaluates them exactly over
//! [`puremp::Rational`] (integers are rationals with denominator one):
//! `+ - * / % **`, parentheses, unary minus, decimal / `0x` / `0b` / `0o`
//! literals, `name = expr` bindings, and function calls
//! (`gcd lcm abs floor ceil num den isqrt fact`). `/` is exact division; `%` is
//! integer remainder. Meta-commands start with `:` (`:help`, `:vars`, `:base`,
//! `:quit`).
//!
//! ```text
//! $ puremp
//! puremp> 2 ** 100
//! 1267650600228229401496703205376
//! puremp> 1/3 + 1/6
//! 1/2
//! puremp> gcd(1071, 462)
//! 21
//! puremp> :base 16
//! puremp> 255
//! ff
//! ```

use std::collections::BTreeMap;
use std::io::{self, BufRead, Write};

use puremp::Rational;

mod eval;

use eval::{Env, eval_line};

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let interactive = is_terminal(&stdin);
    let mut env: Env = BTreeMap::new();
    let mut base: u32 = 10;

    if interactive {
        println!(
            "puremp {} — arbitrary-precision calculator",
            puremp::VERSION
        );
        println!("Type an expression, `name = expr`, or `:help`.");
    }

    let mut lines = stdin.lock().lines();
    loop {
        if interactive {
            let _ = write!(stdout, "puremp> ");
            let _ = stdout.flush();
        }
        let Some(line) = lines.next() else { break };
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix(':') {
            match handle_command(rest.trim(), &env, &mut base) {
                Command::Continue => continue,
                Command::Quit => break,
            }
        }
        match eval_line(trimmed, &mut env) {
            Ok(Some(value)) => println!("{}", format_value(&value, base)),
            Ok(None) => {} // an assignment produced no value to print
            Err(msg) => println!("error: {msg}"),
        }
    }

    if interactive {
        println!();
    }
}

enum Command {
    Continue,
    Quit,
}

fn handle_command(cmd: &str, env: &Env, base: &mut u32) -> Command {
    match cmd {
        "q" | "quit" | "exit" => return Command::Quit,
        "help" | "h" => print_help(),
        "vars" => print_vars(env, *base),
        other => {
            if let Some(arg) = other.strip_prefix("base").map(str::trim) {
                match arg.parse::<u32>() {
                    Ok(b) if (2..=36).contains(&b) => *base = b,
                    _ => println!("usage: :base <radix 2..=36>"),
                }
            } else {
                println!("unknown command `:{other}` (try `:help`)");
            }
        }
    }
    Command::Continue
}

/// Formats a value: integers in the current output radix; non-integers as a
/// reduced `n/d` fraction (base 10).
fn format_value(value: &Rational, base: u32) -> String {
    if value.is_integer() {
        let mut out = String::new();
        // Int::write_radix never fails writing into a String.
        let _ = value.numerator().write_radix(&mut out, base);
        out
    } else {
        value.to_string()
    }
}

fn print_help() {
    println!("puremp calculator commands:");
    println!("  <expr>            evaluate exactly (+ - * / % **, parentheses, unary -)");
    println!("  <name> = <expr>   bind a variable");
    println!("  functions         gcd lcm abs floor ceil num den isqrt fact");
    println!("  literals          decimal, 0x.. 0b.. 0o.., and decimals like 1.5");
    println!("  :base <n>         set the integer output radix (2..=36)");
    println!("  :vars             list bound variables");
    println!("  :help             show this help");
    println!("  :quit             exit");
}

fn print_vars(env: &Env, base: u32) {
    if env.is_empty() {
        println!("(no variables bound)");
        return;
    }
    for (name, value) in env {
        println!("{name} = {}", format_value(value, base));
    }
}

/// Best-effort terminal detection so piped input doesn't print a prompt.
fn is_terminal(stdin: &io::Stdin) -> bool {
    use std::io::IsTerminal;
    stdin.is_terminal()
}
