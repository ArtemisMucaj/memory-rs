//! `memory-rs` binary entry point.
//!
//! Initializes logging, parses the CLI, builds the DI container from flags +
//! `config.json`, dispatches to the router, and prints the result. Domain
//! errors are reported to stderr with a non-zero exit code.

use clap::Parser;
use tracing_subscriber::EnvFilter;

use memory_rs::cli::{Cli, Command};
use memory_rs::connector::api::{router, Container, ContainerConfig};
use memory_rs::tui::LogCapture;

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let is_tui = matches!(cli.command, Command::Tui);

    // Logs must never reach the terminal while the TUI owns it — they would
    // print over the render and corrupt it. For `tui`, capture logs in memory
    // (the app surfaces recent warnings on its footer); every other command
    // logs to stderr so stdout stays clean and pipeable. Verbosity follows
    // RUST_LOG (default: warnings).
    let filter = || EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
    let log_capture = LogCapture::new();
    if is_tui {
        tracing_subscriber::fmt()
            .with_env_filter(filter())
            .with_ansi(false)
            .with_writer(log_capture.clone())
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_env_filter(filter())
            .with_writer(std::io::stderr)
            .init();
    }

    let container_config = ContainerConfig {
        data_dir: cli
            .data_dir
            .clone()
            .unwrap_or_else(ContainerConfig::default_data_dir),
        embedding_dimensions: cli.embedding_dimensions,
        openai_endpoint: cli.openai_endpoint.clone(),
    };

    if let Err(e) = run(cli, container_config, log_capture).await {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

async fn run(
    cli: Cli,
    config: ContainerConfig,
    log_capture: LogCapture,
) -> Result<(), memory_rs::DomainError> {
    let container = Container::new(config)?;

    // The TUI takes over the terminal and produces no printable output, so it is
    // dispatched here rather than through the (text-returning) router.
    if matches!(cli.command, Command::Tui) {
        return memory_rs::tui::run(container, log_capture).await;
    }

    let output = router::run(cli, &container).await?;
    print!("{output}");
    if !output.ends_with('\n') {
        println!();
    }
    Ok(())
}
