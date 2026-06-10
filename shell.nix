let
  pkgs = import (fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/d6c71932130818840fc8fe9509cf50be8c64634f.tar.gz";
    sha256 = "1klgyhj98j3gfsql5sn9rapyx62qk5g8adk5zh9mnc4d0fj61gdr";
  }) { config.allowUnfree = true; }; # allowUnfree: CUDA toolkit is unfree
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

    # CUDA — burn's `cuda` backend (cubecl-cuda). cudatoolkit provides
    # nvcc/nvrtc/cudart; libcuda (the driver) is NOT here, it ships with the
    # host NVIDIA driver at /run/opengl-driver/lib (on LD_LIBRARY_PATH below).
    cudaPackages.cudatoolkit
  ];

  # cubecl resolves the CUDA toolkit through CUDA_PATH.
  CUDA_PATH = pkgs.cudaPackages.cudatoolkit;

  # Vulkan ICD + CUDA runtime libs. libcuda.so (driver) is host-only, so append
  # the raw /run/opengl-driver/lib path after the nix-store libs.
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
    vulkan-loader
    udev
    alsa-lib
    libxkbcommon
    wayland
    cudaPackages.cudatoolkit
  ]) + ":/run/opengl-driver/lib";
}
