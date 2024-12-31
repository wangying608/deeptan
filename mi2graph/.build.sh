cargo clean
# cargo build
cargo build --release
# cargo build --release --target=x86_64-pc-windows-gnu

# musl is much slower than gnu
# cargo build --release --target=x86_64-unknown-linux-musl
