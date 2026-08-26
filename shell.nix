let
  pkgs = import (fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/6b5e5b7a6631f065bf6908986990b37d845f847f.tar.gz";
    sha256 = "0vi99516bn335vdzcjmvrkff8ikj0brpmjfcfdrjnb8bfd0wlr5j";
  }) {};
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    cargo
    rustc
    clippy
    rustfmt

    pkg-config
    udev
    alsa-lib
    vulkan-loader
    vulkan-headers
    vulkan-validation-layers

    libx11
    libxcursor
    libxi
    libxrandr

    libxkbcommon
    wayland

    clang
    mold

    ffmpeg

    # The wasm32 web-build toolchain (rl#411 stage 6), pinned HERE so the release
    # builder needs no channel/<nixpkgs>: lld links the wasm, wasm-bindgen-cli must
    # match Cargo.lock's wasm-bindgen (run.sh asserts), and ring's C sources need an
    # UNWRAPPED clang (the nix-wrapped one emits x86 objects) — see the env below.
    lld
    wasm-bindgen-cli
  ];

  CC_wasm32_unknown_unknown = "${pkgs.llvmPackages.clang-unwrapped}/bin/clang";
  AR_wasm32_unknown_unknown = "${pkgs.llvm}/bin/llvm-ar";

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
    vulkan-loader
    udev
    alsa-lib
    libxkbcommon
    wayland
  ]);
}
