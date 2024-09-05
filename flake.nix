{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/master";
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
              ffmpeg-full
              go-task
              parallel
              pre-commit
              pv
              python312
              rsync
              websocat
              uv
              cudaPackages_12.cudnn_8_9
            ];

            # https://github.com/NixOS/nixpkgs/issues/278976#issuecomment-1879685177
            # NOTE: Without adding `/run/...` the following error occurs
            # RuntimeError: CUDA failed with error CUDA driver version is insufficient for CUDA runtime version
            #
            # NOTE: sometimes it still doesn't work but rebooting the system fixes it
            LD_LIBRARY_PATH = "/run/opengl-driver/lib:${
              pkgs.lib.makeLibraryPath [
                pkgs.cudaPackages_12.cudnn_8_9
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
