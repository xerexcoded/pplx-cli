#!/usr/bin/env bash
set -euo pipefail

VAULT_DIR="${1:-$(mktemp -d /tmp/pplx-vault-XXXXXX)}"
mkdir -p "$VAULT_DIR"/{notes,projects,research,daily,journal}

cleanup() {
    if [ "${1:-}" = "cleanup" ]; then
        echo "Cleaning up vault: $VAULT_DIR"
        rm -rf "$VAULT_DIR"
    fi
}

trap 'exit' INT TERM
trap 'cleanup cleanup' EXIT

echo "=== PPLX Knowledge Graph Demo ==="
echo "Vault: $VAULT_DIR"
echo ""

# ── Root index ──
cat > "$VAULT_DIR/index.md" <<'EOF'
# 📚 Knowledge Vault

Welcome to my knowledge base. This is the entry point.

## Active Areas
- [[notes/ideas|Ideas & Concepts]]
- [[projects/work|Work Projects]]
- [[research/machine-learning|ML Research]]

## Quick Links
- [[daily/2026-05-15|Yesterday's Notes]]
- [[journal/goals|Personal Goals]]
EOF

# ── Notes ──
cat > "$VAULT_DIR/notes/ideas.md" <<'EOF'
# 💡 Ideas & Concepts

## Project Ideas
- [[../projects/side-project|Side Project: CLI Tool]]
- [[../projects/work|Work Project: Dashboard]]

## Learning
- [[../research/machine-learning|ML Fundamentals]]
- [[../research/distributed-systems|Distributed Systems]]

## Random Thoughts
- Should explore [Go language](https://go.dev)
- Related to [[../journal/reflections|Personal Reflections]]
EOF

cat > "$VAULT_DIR/notes/architecture.md" <<'EOF'
# 🏗️ Architecture Notes

## System Design
- [[../projects/work|Work System]] uses microservices
- [[../research/distributed-systems|Distributed patterns]]
- [[../notes/ideas|Ideas]] for improvement

## References
- [Clean Architecture](https://blog.cleancoder.com)
- [[../research/system-design-primer|System Design Primer]]
EOF

# ── Projects ──
cat > "$VAULT_DIR/projects/work.md" <<'EOF'
# 💼 Work Project: Dashboard

## Dependencies
- [[../research/machine-learning|ML Pipeline]]
- [[../notes/architecture|System Architecture]]
- [[../research/distributed-systems|Scaling Strategy]]

## Status
- Phase 1: Complete
- Phase 2: In progress (see [[../daily/2026-05-14]])

## Related
- [[side-project|Side Project]] shares some components
EOF

cat > "$VAULT_DIR/projects/side-project.md" <<'EOF'
# 🔧 Side Project: CLI Tool

Built with:
- [Typer](https://typer.tiangolo.com)
- [D3.js](https://d3js.org) for visualization

## Links
- Inspired by [[../notes/ideas|various ideas]]
- Shares patterns with [[work|Work Dashboard]]
- Documented in [[../research/tooling|Tooling Research]]
EOF

# ── Research ──
cat > "$VAULT_DIR/research/machine-learning.md" <<'EOF'
# 🤖 Machine Learning

## Topics
- [[../notes/ideas|Applied Ideas]]
- [[../projects/work|Production ML Pipeline]]

## Resources
- [Transformers Paper](https://arxiv.org/abs/1706.03762)
- [[distributed-systems|Distributed Training]]
- [[../journal/learning-log|Learning Log]]
EOF

cat > "$VAULT_DIR/research/distributed-systems.md" <<'EOF'
# 🌐 Distributed Systems

## Concepts
- [[../notes/architecture|Architecture Patterns]]

## Applications
- [[../projects/work|Work Dashboard Scaling]]
- [[machine-learning|Distributed ML]]

## Papers
- [Dynamo Paper](https://www.allthingsdistributed.com/files/amazon-dynamo-sosp2007.pdf)
EOF

cat > "$VAULT_DIR/research/tooling.md" <<'EOF'
# 🛠️ Tooling Research

## CLI Frameworks
- [Typer](https://typer.tiangolo.com) - Used in [[../projects/side-project|Side Project]]

## Build Systems
- [[system-design-primer|Design Patterns]]
EOF

cat > "$VAULT_DIR/research/system-design-primer.md" <<'EOF'
# 📐 System Design Primer

## References
- [[../notes/architecture|Architecture]]
- [[distributed-systems|Distributed Systems]]

## Resources
- [System Design Primer](https://github.com/donnemartin/system-design-primer)
EOF

# ── Daily Notes ──
cat > "$VAULT_DIR/daily/2026-05-14.md" <<'EOF'
# 📅 2026-05-14

## Tasks
- [x] Review [[../projects/work|Dashboard]] Phase 1
- [x] Read [[../research/distributed-systems|DS Paper]]
- [ ] Start [[../projects/side-project|Side Project]]

## Notes
- Interesting idea from [[../notes/ideas|Ideas page]]
- Need to update [[../notes/architecture|Architecture docs]]
EOF

cat > "$VAULT_DIR/daily/2026-05-15.md" <<'EOF'
# 📅 2026-05-15

## Tasks
- [x] Draft [[../journal/goals|Q3 Goals]]
- [x] Meeting about [[../projects/work|Work Dashboard]]

## Thoughts
- [[../notes/ideas|New feature idea]] for side project
- Should read more about [[../research/tooling|CLI tools]]
EOF

# ── Journal ──
cat > "$VAULT_DIR/journal/goals.md" <<'EOF'
# 🎯 Personal Goals

## Q3 2026
- Ship [[../projects/side-project|Side Project]] v1.0
- Complete [[../research/machine-learning|ML Study]]
- Write about [[../notes/architecture|Architecture]]

## Long Term
- Master [[../research/distributed-systems|Distributed Systems]]
- Build from [[../notes/ideas|Idea Backlog]]
EOF

cat > "$VAULT_DIR/journal/reflections.md" <<'EOF'
# 📝 Reflections

This week:
- Progress on [[../projects/work|Work]]
- Learned from [[../research/machine-learning|ML Research]]
- Inspired by [[../daily/2026-05-15|Yesterday's notes]]

## Misc
- [Personal site](https://example.com)
EOF

cat > "$VAULT_DIR/journal/learning-log.md" <<'EOF'
# 📖 Learning Log

## May 2026
- [[../research/machine-learning|ML Week 1]] - Transformers
- [[../research/distributed-systems|DS Week 2]] - Consensus
- [[../notes/architecture|Architecture Review]]

## April 2026
- Started [[../projects/side-project|Side Project]]
EOF

# ── Stats ──
echo "Files created:"
echo ""
find "$VAULT_DIR" -name "*.md" | sort | while read -r f; do
    relative="${f#$VAULT_DIR/}"
    links=$(grep -c '\[\[' "$f" 2>/dev/null || echo 0)
    echo "  $relative ($links wikilinks)"
done

TOTAL_FILES=$(find "$VAULT_DIR" -name "*.md" | wc -l | tr -d ' ')
TOTAL_LINKS=$(grep -r '\[\[' "$VAULT_DIR" --include="*.md" | wc -l | tr -d ' ')
echo ""
echo "Total: $TOTAL_FILES files, $TOTAL_LINKS wikilinks"
echo ""

# ── Launch ──
# Find the CLI command
if command -v perplexity &>/dev/null; then
    PPLX_CMD="perplexity"
elif command -v poetry &>/dev/null; then
    PPLX_CMD="poetry run perplexity"
else
    echo "Error: perplexity CLI not found. Install with: pip install -e ."
    exit 1
fi

echo "Launching knowledge graph server..."
echo ""

$PPLX_CMD knowledge-graph --dir "$VAULT_DIR" --title "Knowledge Vault Demo" 2>&1 || {
    printf "\n\nTo view offline:\n"
    echo "  $PPLX_CMD knowledge-graph --dir '$VAULT_DIR' --output /tmp/kg-demo.html"
    echo "  open /tmp/kg-demo.html"
}
