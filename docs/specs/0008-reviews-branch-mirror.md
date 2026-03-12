# Spec: Mirror PR review output into a `reviews` branch

## Why
PR review bots (e.g. `codex-connector`) post findings on GitHub, but this repo’s local Codex sessions may not reliably access GitHub APIs.
We want a git-native way to pull those findings into the local workspace without copy/paste.

## Goal
- Every PR gets a “review bundle” written to a dedicated branch (`reviews`) as:
  - `docs/reviews/pr-<num>.md` (human-readable)
  - `docs/reviews/pr-<num>.json` (machine-readable)
- Local workflow is always:
  - `git fetch origin reviews`
  - `git show origin/reviews:docs/reviews/pr-<num>.md`

## Non-goals
- No attempt to “apply fixes” automatically.
- No requirement that local environments can reach `github.com` APIs.
- No publishing for PRs from forks (avoid token/permission/security complexity).

## Branch model
- `reviews` is an **orphan branch** (no shared history with `main`).
- It contains only generated artifacts under `docs/reviews/`.
- It is write-only from CI.

## Output format
`docs/reviews/pr-<num>.md` includes:
- PR metadata: title, author, base/head refs, head SHA, updated timestamp.
- Codex-bot findings:
  - Issue comments by bot login(s)
  - PR reviews by bot login(s)
  - PR review comments by bot login(s)
- Optional: Check runs summary for the PR head SHA.

`docs/reviews/pr-<num>.json` includes the same data as structured JSON for downstream tooling.

## Bot login configuration
The workflow reads `REVIEW_BOT_LOGINS` (comma-separated).
Default:
- `codex-connector[bot]`
- `codex-connector`

## Concurrency
Only one run at a time may update `reviews`:
- `concurrency.group: reviews-branch`
- `cancel-in-progress: false`

## GitHub Actions workflow
Create: `.github/workflows/reviews-branch-mirror.yml`

```yaml
name: PR Review Mirror

on:
  pull_request:
    types: [opened, synchronize, reopened, edited, ready_for_review]

permissions:
  contents: write
  pull-requests: read
  issues: read
  checks: read

concurrency:
  group: reviews-branch
  cancel-in-progress: false

env:
  REVIEW_BOT_LOGINS: codex-connector[bot],codex-connector

jobs:
  ci:
    name: CI (lightweight)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      # Keep this cheap: no heavyweight GPU deps, no full `requirements.txt` install.
      - name: Syntax check
        run: python -m compileall -q .
      # Optional: lint/typecheck can be added later without installing torch/cuda wheels.
      # - name: Ruff
      #   run: |
      #     python -m pip install -U ruff
      #     ruff check .
      # - name: Pyright
      #   run: |
      #     python -m pip install -U pyright
      #     pyright

  publish_review:
    name: Publish review bundle to `reviews`
    runs-on: ubuntu-latest
    needs: [ci]
    if: github.event.pull_request.head.repo.full_name == github.repository
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Gather PR review data
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require("fs");
            const path = require("path");
            const botLogins = (process.env.REVIEW_BOT_LOGINS || "")
              .split(",")
              .map(s => s.trim())
              .filter(Boolean);

            const { owner, repo } = context.repo;
            const pr = context.payload.pull_request;
            const prNumber = pr.number;
            const headSha = pr.head.sha;

            const issueComments = await github.paginate(
              github.rest.issues.listComments,
              { owner, repo, issue_number: prNumber, per_page: 100 },
            );
            const reviews = await github.paginate(
              github.rest.pulls.listReviews,
              { owner, repo, pull_number: prNumber, per_page: 100 },
            );
            const reviewComments = await github.paginate(
              github.rest.pulls.listReviewComments,
              { owner, repo, pull_number: prNumber, per_page: 100 },
            );

            let checkRuns = [];
            try {
              const res = await github.rest.checks.listForRef({ owner, repo, ref: headSha, per_page: 100 });
              checkRuns = res.data.check_runs || [];
            } catch (e) {
              // checks permission may not be available in all org settings; keep optional.
              checkRuns = [];
            }

            const byBot = (items) =>
              items.filter((x) => botLogins.length === 0 ? false : botLogins.includes(x.user?.login));

            const payload = {
              generated_at: new Date().toISOString(),
              pr: {
                number: prNumber,
                title: pr.title,
                url: pr.html_url,
                author: pr.user?.login || "",
                base: { ref: pr.base.ref, sha: pr.base.sha },
                head: { ref: pr.head.ref, sha: headSha },
                draft: !!pr.draft,
              },
              bot_logins: botLogins,
              issue_comments: byBot(issueComments).map(c => ({
                id: c.id,
                created_at: c.created_at,
                updated_at: c.updated_at,
                url: c.html_url,
                user: c.user?.login || "",
                body: c.body || "",
              })),
              reviews: byBot(reviews).map(r => ({
                id: r.id,
                submitted_at: r.submitted_at,
                state: r.state,
                url: r.html_url,
                user: r.user?.login || "",
                body: r.body || "",
              })),
              review_comments: byBot(reviewComments).map(rc => ({
                id: rc.id,
                created_at: rc.created_at,
                updated_at: rc.updated_at,
                url: rc.html_url,
                user: rc.user?.login || "",
                path: rc.path || "",
                position: rc.position ?? null,
                line: rc.line ?? null,
                side: rc.side ?? "",
                body: rc.body || "",
              })),
              check_runs: checkRuns.map(cr => ({
                name: cr.name,
                status: cr.status,
                conclusion: cr.conclusion,
                details_url: cr.details_url,
                started_at: cr.started_at,
                completed_at: cr.completed_at,
              })),
              ci: { result: "${{ needs.ci.result }}" },
            };

            const outDir = path.join(process.cwd(), "build", "reviews");
            fs.mkdirSync(outDir, { recursive: true });

            fs.writeFileSync(
              path.join(outDir, `pr-${prNumber}.json`),
              JSON.stringify(payload, null, 2),
              "utf-8",
            );

            const lines = [];
            lines.push(`# PR #${prNumber}: ${pr.title}`);
            lines.push("");
            lines.push(`- URL: ${pr.html_url}`);
            lines.push(`- Author: ${payload.pr.author}`);
            lines.push(`- Base: ${payload.pr.base.ref} @ ${payload.pr.base.sha}`);
            lines.push(`- Head: ${payload.pr.head.ref} @ ${payload.pr.head.sha}`);
            lines.push(`- Draft: ${payload.pr.draft}`);
            lines.push(`- Generated: ${payload.generated_at}`);
            lines.push(`- Bot logins: ${botLogins.join(", ") || "(none configured)"}`);
            lines.push("");

            const section = (title, items, render) => {
              lines.push(`## ${title}`);
              if (!items.length) {
                lines.push("");
                lines.push("_None._");
                lines.push("");
                return;
              }
              lines.push("");
              for (const item of items) {
                lines.push(render(item));
                lines.push("");
              }
            };

            section(
              "Bot issue comments",
              payload.issue_comments,
              (c) => `- ${c.updated_at || c.created_at} (${c.user})\\n  - ${c.url}\\n\\n\\n${c.body}`.trim(),
            );

            section(
              "Bot PR reviews",
              payload.reviews,
              (r) => `- ${r.submitted_at || ""} (${r.user}) [${r.state}]\\n  - ${r.url}\\n\\n\\n${r.body}`.trim(),
            );

            section(
              "Bot PR review comments",
              payload.review_comments,
              (rc) => `- ${rc.updated_at || rc.created_at} (${rc.user})\\n  - ${rc.url}\\n  - ${rc.path}${rc.line ? `:${rc.line}` : ""}\\n\\n\\n${rc.body}`.trim(),
            );

            lines.push("## Check runs (head SHA)");
            lines.push("");
            if (!payload.check_runs.length) {
              lines.push("_None or unavailable._");
              lines.push("");
            } else {
              for (const cr of payload.check_runs) {
                lines.push(`- ${cr.name}: ${cr.status} / ${cr.conclusion || "n/a"}`);
              }
              lines.push("");
            }

            fs.writeFileSync(
              path.join(outDir, `pr-${prNumber}.md`),
              lines.join("\\n"),
              "utf-8",
            );

      - name: Update `reviews` branch
        run: |
          set -euo pipefail

          PR_NUM="${{ github.event.pull_request.number }}"
          SRC_DIR="build/reviews"

          git config user.name "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

          git fetch origin reviews || true
          if git show-ref --verify --quiet refs/remotes/origin/reviews; then
            git checkout -B reviews origin/reviews
          else
            git checkout --orphan reviews
            find . -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
          fi

          mkdir -p docs/reviews
          cat > docs/reviews/README.md <<'EOF'
          # PR Review Bundles

          This branch is generated by CI. It mirrors PR review bot output into git-tracked files so offline/local tooling can read it.

          Local usage:
          - `git fetch origin reviews`
          - `git show origin/reviews:docs/reviews/pr-123.md`
          - `git show origin/reviews:docs/reviews/pr-123.json`
          EOF

          cp -f "${SRC_DIR}/pr-${PR_NUM}.md" "docs/reviews/pr-${PR_NUM}.md"
          cp -f "${SRC_DIR}/pr-${PR_NUM}.json" "docs/reviews/pr-${PR_NUM}.json"

          python - <<'PY'
          import glob
          import os
          files = sorted(glob.glob("docs/reviews/pr-*.md"))
          lines = ["# Index", ""]
          for f in files:
              name = os.path.basename(f)
              pr = name.removeprefix("pr-").removesuffix(".md")
              lines.append(f"- [{name}](./{name})")
          lines.append("")
          open("docs/reviews/index.md","w",encoding="utf-8").write("\\n".join(lines))
          PY

          git add docs/reviews
          if git diff --cached --quiet; then
            echo "No changes to publish."
            exit 0
          fi

          git commit -m "reviews: PR #${PR_NUM} mirror"
          git push origin reviews
```

## Local usage (no checkout required)
```bash
git fetch origin reviews
git show origin/reviews:docs/reviews/pr-123.md
```

