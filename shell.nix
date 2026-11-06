let
  pkgs = import (fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/6b5e5b7a6631f065bf6908986990b37d845f847f.tar.gz";
    sha256 = "0vi99516bn335vdzcjmvrkff8ikj0brpmjfcfdrjnb8bfd0wlr5j";
  }) {};
  # Must equal Cargo.lock's wasm-bindgen (game-web/run.sh asserts); the pin above
  # ships older CLIs only, so this builds the matching one with its own builder.
  wasm-bindgen-cli = pkgs.buildWasmBindgenCli rec {
    src = pkgs.fetchCrate {
      pname = "wasm-bindgen-cli";
      version = "0.2.130";
      hash = "sha256-9Og1O/6/VC3sc5hBjoIWBErmLW2UHGMsTQO+6+m/dng=";
    };
    cargoDeps = pkgs.rustPlatform.fetchCargoVendor {
      inherit src;
      inherit (src) pname version;
      hash = "sha256-j81sUf1m5yXU4FJfGj7GV+hdqVCV8mRmNb0QIsMCjqw=";
    };
  };
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

    # The wasm32 web-build toolchain (rl#411 stage 6), pinned HERE so a web build
    # needs no channel/<nixpkgs>: lld links the wasm, wasm-bindgen-cli must
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

  passthru.nixpkgs = toString pkgs.path;
}
