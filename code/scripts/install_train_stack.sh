#!/bin/bash
# Build the training stack of one environment that already holds rfs.
#
#   scripts/install_train_stack.sh <DATA_ROOT>/envs/paperimage [repo-checkout]
#
# The whole set is pinned in requirements-train.txt; this script only adds the
# two things a plain `pip install -r` cannot know: which interpreter to use, and
# that the editable rfs under `repo-checkout` must be reinstalled so `judge
# train` runs against the checkout that was just deployed.
set -euo pipefail

ENV_PREFIX="${1:?usage: install_train_stack.sh <env-prefix> [repo-checkout]}"
CHECKOUT="${2:-}"
PY="$ENV_PREFIX/bin/python"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r "$ROOT/requirements-train.txt"

if [ -n "$CHECKOUT" ]; then
  "$PY" -m pip install -e "$CHECKOUT"
fi

"$PY" - <<'PYEOF'
import importlib
for name in ("torch", "torchvision", "transformers", "peft", "bitsandbytes", "accelerate"):
    module = importlib.import_module(name)
    print("ok", name, getattr(module, "__version__", "?"))
import torch
print("cuda", torch.cuda.is_available(), torch.version.cuda)
PYEOF

echo "training stack ready in $ENV_PREFIX"
