{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixpkgs-master.url = "github:NixOS/nixpkgs/master";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs =
    {
      nixpkgs,
      nixpkgs-master,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        pkgs-master = import nixpkgs-master {
          inherit system;
          config.allowUnfree = true;
        };

        linuxOnlyPkgs =
          with pkgs;
          if system == "x86_64-linux" then
            [
              cudaPackages_12.cudnn
              cudaPackages_12.libcublas
              cudaPackages_12.libcurand
              cudaPackages_12.libcufft
              cudaPackages_12.cuda_cudart
              cudaPackages_12.cuda_nvrtc
            ]
          else
            [ ];

        # https://github.com/nixos/nixpkgs/issues/278976#issuecomment-1879685177
        # NOTE: Without adding `/run/...` the following error occurs
        # RuntimeError: CUDA failed with error CUDA driver version is insufficient for CUDA runtime version
        #
        # NOTE: sometimes it still doesn't work but rebooting the system fixes it
        # TODO: check if `LD_LIBRARY_PATH` needs to be set on MacOS
        linuxLibPath =
          if system == "x86_64-linux" then
            "/run/opengl-driver/lib:${
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
            }"
          else
            "";

      in
      {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs =
            with pkgs;
            [
              act
              docker
              docker-compose
              ffmpeg-full
              go-task
              grafana-loki
              parallel
              pv
              python312
              tempo
              uv
              websocat
            ]
            ++ linuxOnlyPkgs;

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
