{
  inputs = {
    mozpkgs = {
      url = "github:mozilla/nixpkgs-mozilla";
      flake = false;
    };
    naersk = {
      url = "github:nmattia/naersk";
    };
    nixpkgs = {
      url = "github:nixOS/nixpkgs/6b1b72c0f887a478a5aac355674ff6df0fc44f44";
    };
    nixpkgs-flatbuffers = {
      url = "github:nixOS/nixpkgs/d1c3fea7ecbed758168787fe4e4a3157e52bc808";
    };
    utils = {
      url = "github:numtide/flake-utils";
    };
  };

  outputs = { self, naersk, mozpkgs, nixpkgs, nixpkgs-flatbuffers, utils }:
  utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs { inherit system; };
      flatbuffers = (import nixpkgs-flatbuffers { inherit system; }).flatbuffers;
      fixed-tensorflow-lite = (pkgs.tensorflow-lite
        .override { inherit flatbuffers; })
        .overrideAttrs ( self: super: { meta.knownVulnerabilities = []; }
      );
      stdenv = pkgs.clangStdenv;

      mozilla = pkgs.callPackage (mozpkgs + "/package-set.nix") {};
      rust = (mozilla.rustChannelOf {
        date = "2023-05-07";
        channel = "nightly";
        sha256 = "sha256-t7DNlUBS9R7PphCxOU7ITXx1DGEhDOca0Q+Kyt7NHMA=";
      }).rust;
      naersk-lib = pkgs.callPackage naersk {
        cargo = rust;
        rustc = rust;
      };
      pname = "reducedemb";
    in {
      defaultPackage = naersk-lib.buildPackage {
        name = pname;
        version = "0.1.0";
        root = ./.;
        src = ./.;
        doCheck = true; # run the tests (nix logs to view output logs)
        LIBCLANG_PATH="${pkgs.libclang.lib}/lib";
	#mode = "clippy";
        TFLITE_X86_64_LIB_DIR="${fixed-tensorflow-lite}/lib";
        TFLITE_LIB_DIR="${fixed-tensorflow-lite}/lib";
        RUST_BACKTRACE=1;
        gitSubmodules = true;
        #singleStep = true;
        preConfigure = ''
          # Set C flags for Rust's bindgen program. Unlike ordinary C
              # compilation, bindgen does not invoke $CC directly. Instead it
              # uses LLVM's libclang. To make sure all necessary flags are
              # included we need to look in a few places.
              # TODO: generalize this process for other use-cases.
              export BINDGEN_EXTRA_CLANG_ARGS="$(< ${stdenv.cc}/nix-support/libc-crt1-cflags) \
                $(< ${stdenv.cc}/nix-support/libc-cflags) \
                $(< ${stdenv.cc}/nix-support/cc-cflags) \
                $(< ${stdenv.cc}/nix-support/libcxx-cxxflags) \
                ${pkgs.lib.optionalString stdenv.cc.isClang "-idirafter ${stdenv.cc.cc}/lib/clang/${pkgs.lib.getVersion stdenv.cc.cc}/include"} \
                ${pkgs.lib.optionalString stdenv.cc.isGNU "-isystem ${stdenv.cc.cc}/include/c++/${pkgs.lib.getVersion stdenv.cc.cc} -isystem ${stdenv.cc.cc}/include/c++/${pkgs.lib.getVersion stdenv.cc.cc}/${stdenv.hostPlatform.config} -idirafter ${stdenv.cc.cc}/lib/gcc/${stdenv.hostPlatform.config}/${pkgs.lib.getVersion stdenv.cc.cc}/include"}
              "
        '';
        buildInputs = with pkgs; [
          vtk
          opencv
          fixed-tensorflow-lite
        ];
        nativeBuildInputs = with pkgs; [
          #breakpointHook
          stdenv.cc
          libclang
          pkgconfig
        ];
      };
    });
}
