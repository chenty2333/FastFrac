#!/bin/sh
# Download the SNAP inputs into this directory and verify their SHA-256.
set -e
cd "$(dirname "$0")"
B=https://snap.stanford.edu/data
for p in email-Eu-core.txt.gz ca-GrQc.txt.gz ca-CondMat.txt.gz roadNet-PA.txt.gz roadNet-CA.txt.gz \
         web-Google.txt.gz as-skitter.txt.gz bigdata/communities/com-amazon.ungraph.txt.gz \
         bigdata/communities/com-dblp.ungraph.txt.gz bigdata/communities/com-youtube.ungraph.txt.gz \
         bigdata/communities/com-lj.ungraph.txt.gz; do
  f=$(basename "$p"); [ -f "$f" ] || curl -fL -o "$f" "$B/$p"
done
sha256sum -c SHA256SUMS
