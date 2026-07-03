//! `puremp` — a command-line arbitrary-precision calculator / REPL.
//!
//! Reads expressions on standard input and evaluates them over [`puremp::Int`]:
//! `+ - * / % **`, parentheses, unary minus, decimal literals, and `name = expr`
//! variable bindings. `/` and `%` are truncated (round-toward-zero) integer
//! division. Meta-commands start with `:` (`:help`, `:vars`, `:quit`).
//!
//! ```text
//! $ puremp
//! puremp> 2 ** 100
//! 1267650600228229401496703205376
//! puremp> x = 1000
//! puremp> x * x - 1
//! 999999
//! ```

use std::collections::BTreeMap;
use std::io::{self, BufRead, Write};

mod eval;

use eval::{Env, eval_line};

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let interactive = is_terminal(&stdin);
    let mut env: Env = BTreeMap::new();

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
            match rest.trim() {
                "q" | "quit" | "exit" => break,
                "help" | "h" => print_help(),
                "vars" => print_vars(&env),
                other => println!("unknown command `:{other}` (try `:help`)"),
            }
            continue;
        }
        match eval_line(trimmed, &mut env) {
            Ok(Some(value)) => println!("{value}"),
            Ok(None) => {} // an assignment produced no value to print
            Err(msg) => println!("error: {msg}"),
        }
    }

    if interactive {
        println!();
    }
}

fn print_help() {
    println!("puremp calculator commands:");
    println!("  <expr>            evaluate (operators: + - * / % **, parentheses, unary -)");
    println!("  <name> = <expr>   bind a variable");
    println!("  :vars             list bound variables");
    println!("  :help             show this help");
    println!("  :quit             exit");
}

fn print_vars(env: &Env) {
    if env.is_empty() {
        println!("(no variables bound)");
        return;
    }
    for (name, value) in env {
        println!("{name} = {value}");
    }
}

/// Best-effort terminal detection so piped input doesn't print a prompt.
///
/// `std::io::IsTerminal` is stable since Rust 1.70; the crate's MSRV is 1.88.
fn is_terminal(stdin: &io::Stdin) -> bool {
    use std::io::IsTerminal;
    stdin.is_terminal()
}
