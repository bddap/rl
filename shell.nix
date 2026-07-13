let
  pkgs = import (fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/e7a3ca8092b61ff85b6a45bf863ea2b2d6a661b3.tar.gz";
    sha256 = "1h4jkfjbdp9y0alp86z38g60mqw7rzx89gn16dbvw8wn2z7r002j";
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
  ];

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
    vulkan-loader
    udev
    alsa-lib
    libxkbcommon
    wayland
  ]);
}
