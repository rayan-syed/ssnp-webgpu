#!/bin/bash

# Check if rustup is already installed
if command -v rustc &> /dev/null; then
    echo "Rust is already installed."
    rustc --version
    exit 0
fi

# Install Rust using rustup
echo "Installing Rust..."
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Add cargo to PATH for the current session
source $HOME/.cargo/env

# Verify installation
if command -v rustc &> /dev/null; then
    echo "Rust has been installed successfully."
    rustc --version
else
    echo "Rust installation failed."
    exit 1
fi