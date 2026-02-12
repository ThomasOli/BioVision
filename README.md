# BioVision

## Development

Install dependencies and run the app locally:

```bash
npm install
npm run dev
```

## Build

Create production build artifacts:

```bash
npm run build
```

## Multi-Platform Packaging

Generate installers/packages locally:

```bash
npm run dist
```

Platform-specific packaging:

```bash
npm run dist:win
npm run dist:mac
npm run dist:linux
```

## Releases

This repo uses GitHub Actions (`.github/workflows/release.yml`) to publish cross-platform release assets.

Release trigger:
- A push of a tag matching `v*` (example: `v0.1.0`)

Release flow after features are merged:
1. Merge feature branch into `main`.
2. Pull latest `main`.
3. Create and push a version tag on that commit.

```bash
git checkout main
git pull
git tag v0.1.0
git push origin v0.1.0
```

What happens next:
- GitHub Actions builds Windows, macOS, and Linux artifacts.
- `electron-builder` publishes those artifacts to GitHub Releases.

## Required Repository Secret

Set this in GitHub repository settings (`Settings -> Secrets and variables -> Actions`):

- `GH_TOKEN`: GitHub Personal Access Token (classic, `repo` scope recommended)

## Electron Builder Config Notes

The publish target is configured in `electron-builder.json5`:
- `publish.owner`: your GitHub username or org name
- `publish.repo`: repository name

