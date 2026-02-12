# BioVision

## Development

Install dependencies and run the app locally:

```bash
npm install
npm run dev
```

Python backend in development runs from source files in `backend/*.py` and uses your local Python.

## Build

Create production frontend/electron build artifacts:

```bash
npm run build
```

## Multi-Platform Packaging

Install Python build dependencies once:

```bash
python -m pip install -r backend/requirements.txt pyinstaller
```

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

`dist*` commands compile the Python backend into executables and bundle them into the installer under `resources/python`.

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
- `electron-builder` publishes those artifacts to GitHub Releases using `${{ github.token }}`.

## Electron Builder Config Notes

The publish target is configured in `electron-builder.json5`:
- `publish.owner`: your GitHub username or org name
- `publish.repo`: repository name

Use names only, not full URLs.
