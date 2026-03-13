#!/bin/bash

echo ""
echo "=================================================="
echo "         🚁 ArcDrone Environment Setup            "
echo "=================================================="
echo ""

# ── GitHub Login ──────────────────────────────────────
echo "🔑 GitHub Login (HTTPS with PAT)"
echo "👉 If you don't have a PAT: https://github.com/settings/tokens"
echo "   - Scope: 'repo'"
echo "   - Expiry: 30-90 days recommended"
echo ""
gh auth login --hostname github.com --git-protocol https --web=false

echo ""

# ── wandb Login ───────────────────────────────────────
echo "📊 Weights & Biases Login"
echo "👉 Find your API key here: https://wandb.ai/authorize"
echo ""
wandb login

echo ""
echo "=================================================="
echo "✅ All set! You're in /workspace"
echo "   Clone your repo with:"
echo "   gh repo clone your-username/arcdrone"
echo "=================================================="
echo ""