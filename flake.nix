{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs =
    { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
      in
      {
        devShells = {
          default = pkgs.mkShell {
            nativeBuildInputs = with pkgs; [
              act
              cudaPackages_12.cudnn
              cudaPackages_12.libcublas
              ffmpeg-full
              go-task
              grafana-loki
              parallel
              pv
              python312
              tempo
              uv
              websocat
            ];

            # https://github.com/NixOS/nixpkgs/issues/278976#issuecomment-1879685177
            # NOTE: Without adding `/run/...` the following error occurs
            # RuntimeError: CUDA failed with error CUDA driver version is insufficient for CUDA runtime version
            #
            # NOTE: sometimes it still doesn't work but rebooting the system fixes it
            LD_LIBRARY_PATH = "/run/opengl-driver/lib:${
              pkgs.lib.makeLibraryPath [
                # Needed for `faster-whisper`
                pkgs.cudaPackages_12.cudnn
                pkgs.cudaPackages_12.libcublas
                # The 4 cuda packages below are needed for `onnxruntime-gpu`
                pkgs.cudaPackages_12.libcurand
                pkgs.cudaPackages_12.libcufft
                pkgs.cudaPackages_12.cuda_cudart
                pkgs.cudaPackages_12.cuda_nvrtc

                # Needed for `soundfile`
                pkgs.portaudio

                pkgs.zlib
                pkgs.stdenv.cc.cc
                pkgs.openssl
              ]
            }";

            shellHook = ''
              source .venv/bin/activate
              source .env
            '';
          };
        };
        formatter = pkgs.nixfmt;
      }
    );
}
