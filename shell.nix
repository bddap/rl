let
  pkgs = import (fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/d6c71932130818840fc8fe9509cf50be8c64634f.tar.gz";
    sha256 = "1klgyhj98j3gfsql5sn9rapyx62qk5g8adk5zh9mnc4d0fj61gdr";
  }) {};
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    # Rust toolchain
    cargo
    rustc
    clippy
    rustfmt

    # Bevy system dependencies
    pkg-config
    udev
    alsa-lib
    vulkan-loader
    vulkan-headers
    vulkan-validation-layers

    # X11
    libx11
    libxcursor
    libxi
    libxrandr

    # Wayland
    libxkbcommon
    wayland

    # Build tools
    clang
    mold
  ];

  # Point Vulkan ICD loader at the right drivers
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
    vulkan-loader
    udev
    alsa-lib
    libxkbcommon
    wayland
  ]);
}
