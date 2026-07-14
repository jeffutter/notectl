{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
    rust-overlay.inputs.nixpkgs.follows = "nixpkgs";
    crane.url = "github:ipetkov/crane";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      rust-overlay,
      crane,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };

        lib = nixpkgs.lib;
        craneLib = crane.mkLib pkgs;

        src = lib.cleanSourceWith { src = craneLib.path ./.; };

        # Single source of truth for the Rust toolchain (rustc/cargo/rustfmt/clippy
        # all from the same rust-overlay release) — nixpkgs' own `cargo`/`rustc`
        # are deliberately not used alongside it, since mixing the two pulls in two
        # differently-versioned toolchains with no defined precedence between them.
        rustToolchain = pkgs.rust-bin.stable.latest.default;

        envVars =
          { }
          // (lib.attrsets.optionalAttrs pkgs.stdenv.isLinux {
            RUSTFLAGS = "-Clinker=clang -Clink-arg=--ld-path=${pkgs.mold}/bin/mold";
          });

        commonArgs = (
          {
            inherit src;
            buildInputs =
              with pkgs;
              [
                clang
                rustToolchain
              ]
              ++ lib.optionals stdenv.isDarwin [ libiconv ];
          }
          // envVars
        );
        cargoArtifacts = craneLib.buildDepsOnly commonArgs;

        notectl = craneLib.buildPackage (
          commonArgs
          // {
            inherit cargoArtifacts;
            cargoExtraArgs = "--bin notectl";
          }
        );

        notectl-remote = craneLib.buildPackage (
          commonArgs
          // {
            inherit cargoArtifacts;
            cargoExtraArgs = "--bin notectl-remote";
          }
        );
      in
      with pkgs;
      {
        packages = {
          default = notectl;
          inherit notectl notectl-remote;
        };

        devShells = {
          # Full local-dev shell: compiler toolchain plus editor/workflow tooling.
          default = mkShell (
            {
              packages = [
                cargo-audit
                cargo-nextest
                cargo-watch
                clang
                lefthook
                rust-analyzer
                rustToolchain
              ];
            }
            // envVars
          );

          # Lean CI shell: compiler toolchain only, no editor/dev-workflow tools,
          # so CI jobs realize a smaller, unambiguous closure.
          ci = mkShell (
            {
              packages = [
                clang
                rustToolchain
              ]
              ++ lib.optionals stdenv.isDarwin [ libiconv ];
            }
            // envVars
          );
        };

        formatter = nixpkgs-fmt;
      }
    );
}
