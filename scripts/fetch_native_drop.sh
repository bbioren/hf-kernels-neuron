#!/usr/bin/env bash
# Fetch the Native PyTorch beta drop onto trn2, without putting AWS credentials on the host.
#
# WHY PRESIGNED URLS
# The instance role (EpoxyChronicleInstanceRole) has no S3 permissions at all — `s3:ListAllMyBuckets`
# is denied "because no identity-based policy allows" it. So the instance cannot read the bucket even
# though Samir allowlisted our account on it: cross-account S3 needs BOTH a bucket policy allow and
# an identity policy allow, and we only have the first.
#
# The obvious workaround — copy our Isengard credentials to the instance — puts short-lived admin
# credentials on a remote host for no good reason. Presigned URLs carry a scoped, time-limited
# signature for ONE object instead, so nothing reusable leaves the laptop.
#
# Skips rpm/ (this is Ubuntu) and downloads deb/ and pip/ only.
#
# Usage (needs valid creds for the account allowlisted on the bucket):
#   ada credentials update --account=<AWS_ACCOUNT_ID> --provider=isengard --role=Admin --once
#   ./scripts/fetch_native_drop.sh
#
# Note: no ReadOnly role exists in that account, hence Admin. --once keeps it to a single session
# rather than writing a long-lived profile.
set -euo pipefail

BUCKET="huggingface-aws"
PREFIX="pytorch-native/drop_jun_25"
HOST="${HOST:-trn2}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/native_drop}"
EXPIRY=3600
# The bucket lives in ap-northeast-2. This MUST be passed explicitly: `aws s3 ls` follows S3's
# cross-region redirect transparently, but `presign` bakes the regional endpoint into the URL, so a
# presign made against the CLI's default region returns
#   400 IllegalLocationConstraintException: The ap-northeast-2 location constraint is incompatible
#       for the region specific endpoint this request was sent to
# which reads like a permissions problem and is not one. get-bucket-location is denied to us, so the
# region cannot be discovered — it is recorded here instead.
REGION="${REGION:-ap-northeast-2}"

echo "listing s3://${BUCKET}/${PREFIX}/ (region ${REGION}) ..."
KEYS=$(aws s3 ls "s3://${BUCKET}/${PREFIX}/" --recursive --region "$REGION" \
       | awk '{print $4}' \
       | grep -E '/(deb|pip)/' )

if [ -z "$KEYS" ]; then
    echo "no deb/ or pip/ objects found — check credentials and the prefix"
    exit 1
fi

echo "presigning $(echo "$KEYS" | wc -l | tr -d ' ') object(s), ${EXPIRY}s expiry ..."
URLFILE=$(mktemp)
trap 'rm -f "$URLFILE"' EXIT
for key in $KEYS; do
    sub=$(dirname "$key" | sed "s|^${PREFIX}/||")     # deb or pip
    name=$(basename "$key")
    url=$(aws s3 presign "s3://${BUCKET}/${key}" --expires-in "$EXPIRY" --region "$REGION")
    printf '%s\t%s\t%s\n' "$sub" "$name" "$url" >> "$URLFILE"
done

echo "sending URL list to ${HOST} ..."
scp -q "$URLFILE" "${HOST}:/tmp/native_drop_urls.tsv"

echo "downloading on ${HOST} -> ${REMOTE_DIR} ..."
ssh "$HOST" "mkdir -p ${REMOTE_DIR}/deb ${REMOTE_DIR}/pip && \
  while IFS=\$'\t' read -r sub name url; do \
    out=\"${REMOTE_DIR}/\$sub/\$name\"; \
    if [ -s \"\$out\" ]; then echo \"  have  \$name\"; continue; fi; \
    echo \"  get   \$name\"; \
    curl -fsSL --retry 3 -o \"\$out\" \"\$url\" || { echo \"  FAILED \$name\"; exit 1; }; \
  done < /tmp/native_drop_urls.tsv; \
  rm -f /tmp/native_drop_urls.tsv"

echo
ssh "$HOST" "echo '=== downloaded ==='; ls -lh ${REMOTE_DIR}/deb ${REMOTE_DIR}/pip | grep -v '^total'; \
  echo; echo '=== total ==='; du -sh ${REMOTE_DIR}"
echo
echo "NEXT: create the venv and pip install from ${REMOTE_DIR}/pip (additive, safe)."
echo "      The deb/ install replaces the HOST Neuron runtime and driver — separate decision."
