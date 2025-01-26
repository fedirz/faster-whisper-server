{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        linuxOnlyPkgs = with pkgs; if system == "x86_64-linux" then [
          cudaPackages_12.cudnn
          cudaPackages_12.libcublas
          cudaPackages_12.libcurand
          cudaPackages_12.libcufft
          cudaPackages_12.cuda_cudart
          cudaPackages_12.cuda_nvrtc
        ] else [];

        # LD_LIBRARY_PATH seulement sur Linux
        linuxLibPath = if system == "x86_64-linux" then "/run/opengl-driver/lib:${
          pkgs.lib.makeLibraryPath ([
            pkgs.cudaPackages_12.cudnn
            pkgs.cudaPackages_12.libcublas
            pkgs.cudaPackages_12.libcurand
            pkgs.cudaPackages_12.libcufft
            pkgs.cudaPackages_12.cuda_cudart
            pkgs.cudaPackages_12.cuda_nvrtc
            pkgs.portaudio
            pkgs.zlib
            pkgs.stdenv.cc.cc
            pkgs.openssl
          ])
        }" else "";

      in {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = with pkgs; [
            act
            ffmpeg-full
            go-task
            grafana-loki
            parallel
            pv
            python312
            tempo
            uv
            websocat
          ] ++ linuxOnlyPkgs;

          LD_LIBRARY_PATH = linuxLibPath;

          shellHook = ''
            source .venv/bin/activate
            source .env
          '';
        };

        formatter = pkgs.nixfmt;
      }
    );
}