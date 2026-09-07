#!/usr/bin/env bash
# Record where Neper, Gmsh and POV-Ray live, on the festim-microstructure env.
#
# Usage (from anywhere, with conda on PATH):
#   tools/link-neper-env.sh                 # neper-env -> festim-microstructure
#   tools/link-neper-env.sh my-neper my-fm  # custom environment names
#
# It sets FM_NEPER_BIN / FM_GMSH_BIN / FM_POVRAY_BIN as conda "environment
# variables" of the FEniCSx env (`conda env config vars`), so they are exported
# on `conda activate festim-microstructure` and unset on deactivate. Nothing is
# written to your shell profile. Re-run it if either environment is recreated.
#
# If you would rather not use conda's env-vars mechanism, export the same three
# variables yourself; festim_microstructure._binaries.find_binary reads them.
set -euo pipefail

NEPER_ENV="${1:-neper-env}"
FM_ENV="${2:-festim-microstructure}"

env_prefix() {
    # `conda env list` prints "name  [*]  /path"; the path is the last field
    conda env list | awk -v n="$1" '$1 == n { print $NF }'
}

NEPER_PREFIX="$(env_prefix "$NEPER_ENV")"
FM_PREFIX="$(env_prefix "$FM_ENV")"
if [ -z "$NEPER_PREFIX" ]; then
    echo "error: no conda environment named '$NEPER_ENV'." >&2
    echo "       conda env create -f environment-neper.yml" >&2
    exit 1
fi
if [ -z "$FM_PREFIX" ]; then
    echo "error: no conda environment named '$FM_ENV'." >&2
    echo "       conda env create -f environment.yml" >&2
    exit 1
fi
case "$NEPER_PREFIX" in
    *[[:space:]]*)
        echo "error: '$NEPER_PREFIX' contains whitespace. Neper re-tokenizes its" >&2
        echo "       arguments, so paths with spaces are torn apart. Move or symlink" >&2
        echo "       the environment somewhere without one." >&2
        exit 1 ;;
esac

vars=()
for prog in neper gmsh povray; do
    bin="$NEPER_PREFIX/bin/$prog"
    var="FM_$(echo "$prog" | tr '[:lower:]' '[:upper:]')_BIN"
    if [ -x "$bin" ]; then
        vars+=("$var=$bin")
        echo "  $var=$bin"
    elif [ "$prog" = povray ]; then
        echo "  $var  (povray not installed in $NEPER_ENV; render checks will be skipped)"
    else
        echo "error: $bin not found or not executable." >&2
        exit 1
    fi
done

conda env config vars set -n "$FM_ENV" "${vars[@]}"
echo "  conda deactivate && conda activate $FM_ENV && fm-check"
