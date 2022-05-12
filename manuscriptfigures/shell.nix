let
  commitHash = "9a3277af47a3cbaf855e413e4c9277240c426916";
  tarballUrl = "https://github.com/NixOS/nixpkgs/archive/${commitHash}.tar.gz";
  pkgs = import (fetchTarball tarballUrl) {};
  stdenv = pkgs.stdenv;
in with pkgs; {
  myProject = stdenv.mkDerivation {
    name = "r-env";
    src = if pkgs.lib.inNixShell then null else nix;

    buildInputs = with rPackages; [
      R
      dplyr
      purrr
      magrittr
      ggplot2
      scales
      GADMTools
      geojsonio
      geojson
      protolite
      V8
      reshape2
      zoo
    ];
   shellHook = ''
             printf "\n\nWelcome to a reproducible R shell :)\n\n"
      '';
  };
}
