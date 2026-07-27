//! `memory-rs` binary entry point.
//!
//! Initializes logging, parses the CLI, builds the DI container from flags +
//! `config.json`, dispatches to the router, and prints the result. Domain
//! errors are reported to stderr with a non-zero exit code.

use clap::Parser;

use memory_rs::cli::{Cli, Command};
use memory_rs::connector::api::{router, Container, ContainerConfig};

#[tokio::main]
async fn main() {
    // Logging goes to stderr so command output on stdout stays clean and
    // pipeable. Verbosity is controlled by RUST_LOG (default: warnings).
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    let container_config = ContainerConfig {
        data_dir: cli
            .data_dir
            .clone()
            .unwrap_or_else(ContainerConfig::default_data_dir),
        embedding_dimensions: cli.embedding_dimensions,
        openai_endpoint: cli.openai_endpoint.clone(),
    };

    if let Err(e) = run(cli, container_config).await {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

async fn run(cli: Cli, config: ContainerConfig) -> Result<(), memory_rs::DomainError> {
    let container = Container::new(config)?;

    // The TUI takes over the terminal and produces no printable output, so it is
    // dispatched here rather than through the (text-returning) router.
    if matches!(cli.command, Command::Tui) {
        return memory_rs::tui::run(container).await;
    }

    let output = router::run(cli, &container).await?;
    print!("{output}");
    if !output.ends_with('\n') {
        println!();
    }
    Ok(())
}
