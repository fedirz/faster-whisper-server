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
              (with python311Packages; huggingface-hub)
              ffmpeg-full
              go-task
              lsyncd
              poetry
              pre-commit
              pv
              pyright
              python311
              websocat
            ];
            shellHook = ''
              source $(poetry env info --path)/bin/activate
              export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
              export LD_LIBRARY_PATH=${pkgs.zlib}/lib:$LD_LIBRARY_PATH
            '';
          };
        };
        formatter = pkgs.nixfmt;
      }
    );
}
