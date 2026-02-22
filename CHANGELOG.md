# Changelog

All notable changes to this repository will be documented here.

This project generally follows the spirit of **Keep a Changelog**, with entries grouped by date and type.

## [Unreleased]

### Added
-

### Changed
-

### Fixed
-

---

## [2026-02-22]

### Added
- Professional repository documentation (README) with WSL2 runbook (SITL + Mission Planner + MAVROS + SEANO).
- AI dependency baseline (`requirements.txt`) and verification steps (Torch + Ultralytics).
- Repository governance:
  - `CONTRIBUTING.md`
  - PR template (`.github/pull_request_template.md`)
  - Issue templates (bug report, feature request) + issue configuration (no blank issues, no dead links).
- MIT License.

### Changed
- `.gitignore` updated to ignore local AI artifacts and virtual environments (`.venv_ai/`, `*.pt`, `runs/`, etc.).

### Fixed
- PR template formatting (multi-line, valid markdown; code blocks closed properly).
