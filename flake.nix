{
  description = "Justin Stone portfolio site build and deploy tooling";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
  };

  outputs =
    { nixpkgs, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
      ];
      forAllSystems =
        f:
        nixpkgs.lib.genAttrs systems (
          system:
          f {
            pkgs = import nixpkgs { inherit system; };
          }
        );
    in
    {
      devShells = forAllSystems (
        { pkgs }:
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              ruby_3_3
              bundler
              nodejs_22
              git
              openssh
              rsync
              pkg-config
              libxml2
              libxslt
              vips
              imagemagick
            ];

            shellHook = ''
              export BUNDLE_PATH=.bundle
              echo "Portfolio shell: bundle install, then bundle exec jekyll build"
            '';
          };
        }
      );
    };
}
