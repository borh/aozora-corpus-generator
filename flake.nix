{
  description = "Python development environment for aozora-corpus-generator";

  # Flake inputs
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  # Flake outputs
  outputs =
    { self, nixpkgs }:
    let
      # Systems supported
      allSystems = [
        "x86_64-linux" # 64-bit Intel/AMD Linux
        "aarch64-linux" # 64-bit ARM Linux
        "x86_64-darwin" # 64-bit Intel macOS
        "aarch64-darwin" # 64-bit ARM macOS
      ];

      # Helper to provide system-specific attributes
      forAllSystems =
        f:
        nixpkgs.lib.genAttrs allSystems (
          system:
          f {
            pkgs = import nixpkgs { inherit system; };
          }
        );
    in
    {
      # Development environment output
      devShells = forAllSystems (
        { pkgs }:
        let
          pname = "unidic-novel";
          version = "202308";
          unidicNovel = pkgs.stdenv.mkDerivation {
            inherit pname version;

            src = pkgs.fetchzip {
              url = "https://ccd.ninjal.ac.jp/unidic_archive/2308/${pname}-v${version}.zip";
              name = "${pname}-v${version}.zip";
              sha256 = "sha256-oKgx/u4HMiwIupWyL95zq2rL4oKQC965kY1lycLm2XE=";
              stripRoot = false;
            };

            phases = [
              "unpackPhase"
              "installPhase"
            ];
            installPhase = ''
              cd $pname
              runHook preInstall
              install -d $out/share/mecab/dic/$pname
              install -m 644 dicrc *.def *.bin *.dic $out/share/mecab/dic/$pname
              runHook postInstall
            '';
          };
        in
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.pkg-config
              pkgs.libxml2
              pkgs.uv
              unidicNovel
            ];

            shellHook = ''
              export AOZORA_UNIDIC_DIR=${unidicNovel}/share/mecab/dic/unidic-novel
            '';
          };
        }
      );
    };
}
