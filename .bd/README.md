# bd - Issue Tracking

Welcome to bd! This repository uses **bd** for issue tracking - a modern, AI-native tool designed to live directly in your codebase alongside your code.

## What is bd?

bd is issue tracking that lives in your repo, making it perfect for AI coding agents and developers who want their issues close to their code. No web UI required - everything works through the CLI and integrates seamlessly with git.

**Learn more:** [github.com/signalnine/bd](https://github.com/signalnine/bd)

## Quick Start

### Essential Commands

```bash
# Create new issues
bd create "Add user authentication"

# View all issues
bd list

# View issue details
bd show <issue-id>

# Update issue status
bd update <issue-id> --claim
bd update <issue-id> --status done

# Sync with Dolt remote
bd dolt push
```

### Working with Issues

Issues in bd are:
- **Git-native**: Stored in Dolt database with version control and branching
- **AI-friendly**: CLI-first design works perfectly with AI coding agents
- **Branch-aware**: Issues can follow your branch workflow
- **Always in sync**: Auto-syncs with your commits

## Why bd?

✨ **AI-Native Design**
- Built specifically for AI-assisted development workflows
- CLI-first interface works seamlessly with AI coding agents
- No context switching to web UIs

🚀 **Developer Focused**
- Issues live in your repo, right next to your code
- Works offline, syncs when you push
- Fast, lightweight, and stays out of your way

🔧 **Git Integration**
- Automatic sync with git commits
- Branch-aware issue tracking
- Dolt-native three-way merge resolution

## Get Started

Try bd in your own projects:

```bash
# Install bd
curl -sSL https://raw.githubusercontent.com/signalnine/bd/main/scripts/install.sh | bash

# Initialize in your repo
bd init

# Create your first issue
bd create "Try out bd"
```

## Learn More

- **Documentation**: [github.com/signalnine/bd/docs](https://github.com/signalnine/bd/tree/main/docs)
- **Examples**: [github.com/signalnine/bd/examples](https://github.com/signalnine/bd/tree/main/examples)

---

*bd: Issue tracking that moves at the speed of thought* ⚡
