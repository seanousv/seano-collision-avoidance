# Contributing Guide — SEANO Collision Avoidance

Thank you for contributing to the SEANO collision avoidance module.
This repository is focused on **vision-based collision avoidance** and the supporting **ROS 2 pipeline** required to run it reliably.

## Scope

This repository **only** covers:
- Perception pipeline (camera input, detection, optional tracking)
- Risk assessment / decision logic (avoidance commands)
- Safety layer (timeouts, failsafe behavior)
- Actuation bridge (ROS 2 → MAVROS → ArduPilot / RC override)
- Simulation integration (SITL + Mission Planner runbook)

Please keep the following **out of scope** (use separate repositories):
- Dashboard / UI
- Telemetry backends, cloud services
- Non-CA navigation stack (full autonomy stack, global path planning) unless required for CA demo
- Large datasets and raw videos (use external storage and link)

---

## Quick Start (Developer Setup)

1) Configure git identity **for this repo only** (recommended):
```bash
git config user.name "SEANO | YourName"
git config user.email "seanousv@gmail.com"
````

2. Verify:

```bash
git config --get user.name
git config --get user.email
```

---

## Branching Model

* `main`: stable and demo-ready baseline
* Feature branches: `feature/<short-name>`

  * examples: `feature/actuation-mux`, `feature/risk-evaluator`, `feature/yolo-inference`
* Fix branches: `fix/<short-name>`

  * examples: `fix/mavros-fcu-url`, `fix/failsafe-timeout`

Guideline:

* Keep feature branches small and focused (easier review & fewer conflicts).

---

## Commit Message Convention

Use a clear prefix:

* `docs:` documentation only (README, diagrams, comments)
* `feat:` new feature or capability
* `fix:` bug fix
* `refactor:` refactor without changing behavior
* `test:` add/update tests
* `chore:` housekeeping (formatting, tooling, repo hygiene)

Examples:

* `feat: add command mux for manual/auto`
* `fix: prevent duplicate publishers on auto topics`
* `docs: add WSL2 SITL+MAVROS runbook`
* `chore: add MIT LICENSE`

---

## Pull Request (PR) Checklist

Before opening a PR:

1. Build check (ROS 2):

```bash
cd seano_ca_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

2. Basic runtime sanity (if your changes touch control/bridge):

* MAVROS connects: `/mavros/state` → `connected: true`
* `run_auto_stack.launch.py` starts without fatal errors
* No duplicate publishers on critical topics (use `ros2 topic info -v <topic>`)

3. Keep the PR focused:

* One feature/fix per PR
* Include a short description of:

  * What changed
  * How to run/test
  * Expected behavior

---

## Code Style (Python / ROS 2)

* Prefer small, readable functions with explicit names.
* Avoid magic constants; use ROS parameters where appropriate.
* Log important state transitions (e.g., failsafe activated, mode switched).
* Topic names should remain stable once used by other nodes.
* Keep node files self-contained and executable via `ros2 run`.

Recommended formatting:

* PEP8 style (reasonable line length)
* Use `black` / `ruff` if the team adopts it later (optional)

---

## Repository Hygiene

Do NOT commit:

* `build/`, `install/`, `log/` (colcon artifacts)
* `.venv/`, `.venv_ai/`, `__pycache__/`
* large binaries, dataset dumps, `.bag`, `.mp4`, `.avi` (store externally)

If you accidentally added large files:

* remove them and rewrite history if needed (ask the maintainer first)

---

## Adding Dependencies

### ROS dependencies

* Add required ROS packages in `seano_ca_ws/src/seano_vision/package.xml`
* Prefer ROS apt packages over pip where possible (e.g., cv_bridge)

### Python dependencies (AI)

* Add to `requirements.txt` (WSL-safe)
* Jetson-specific torch installation should be documented (not forced via pip)

---

## Testing Expectations (Minimum)

For changes touching actuation/bridge:

* Confirm RC override topic publishes:

  * `ros2 topic echo /mavros/rc/override`
* Confirm the vehicle responds (SITL + Mission Planner):

  * heading/track changes visibly when commands change

For changes touching perception:

* Confirm stable publish rate on image/detections for several minutes
* Confirm no memory blow-up / runaway logging

---

## How to Submit Changes

Typical workflow:

```bash
git checkout -b feature/<name>

# make changes...
git status
git add <files>
git commit -m "feat: <short summary>"

git push -u origin feature/<name>
```

Then open a PR into `main`.

---

## Communication / Ownership

* If you are modifying a shared interface (topic names, message types, launch behavior),
  leave a note in the PR description and coordinate with the maintainer.
* When in doubt, keep backward compatibility and add a migration note in README.

---

## Contact

Maintainer email:

* `seanousv@gmail.com`
