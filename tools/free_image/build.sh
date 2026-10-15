#!/bin/bash
# Build the free protocol's container image on the run host.
#
# The host's Docker registry mirror cannot serve images, so the base is
# Ubuntu's own cloud rootfs, unpacked in a user namespace. The unpack cannot
# create device nodes or chown to unmapped ids; those errors are expected and
# do not matter to Apptainer. /etc/resolv.conf is a dangling link in the
# rootfs and must be a file for the build, and snapd's void directory must be
# readable.
set -euo pipefail
OUT=${1:-/scratch/images}
HERE=$(cd "$(dirname "$0")" && pwd)
export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-/scratch/.cache/apptainer}
mkdir -p "$OUT" && cd "$OUT"
URL=https://cloud-images.ubuntu.com/minimal/releases/noble/release/ubuntu-24.04-minimal-cloudimg-amd64-root.tar.xz
[ -s noble-root.tar.xz ] || curl -fL -o noble-root.tar.xz "$URL"
rm -rf noble-rootfs && mkdir noble-rootfs
unshare --user --map-root-user tar -xJf noble-root.tar.xz -C noble-rootfs 2>/dev/null || true
unshare --user --map-root-user bash -c '
  rm -f noble-rootfs/etc/resolv.conf
  printf "nameserver 8.8.8.8\n" > noble-rootfs/etc/resolv.conf
  chmod -R u+rwX noble-rootfs/var/lib/snapd'
apptainer build --fakeroot --force noble.sif noble-rootfs/
apptainer build --fakeroot --force glyph-free.sif "$HERE/glyph-free.def"
rm -rf noble-rootfs
ls -la "$OUT/glyph-free.sif"
