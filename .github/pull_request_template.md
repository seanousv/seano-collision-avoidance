# Summary
<!-- Explain what this PR changes and why. Keep it concise. -->

## What changed?
-

## Why?
-

---

# Type of change
<!-- Check ONE primary type. -->
- [ ] feat (new feature)
- [ ] fix (bug fix)
- [ ] refactor (no behavior change)
- [ ] docs (documentation only)
- [ ] test (tests only)
- [ ] chore (tooling/repo hygiene)

---

# Scope / affected areas
<!-- Check all that apply. -->
- [ ] Actuation / Control (teleop, mux, limiter)
- [ ] MAVROS / MAVLink bridge (RC override, FCU link)
- [ ] Simulation (SITL, Mission Planner)
- [ ] Perception (camera input)
- [ ] Detection / Tracking
- [ ] Risk / Decision logic
- [ ] Launch / Config
- [ ] Documentation

---

# How to test
<!-- Provide exact commands used. If not tested, explain why. -->

## ROS 2 build
```bash
cd seano_ca_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
````

## Runtime / demo (if applicable)

```bash
source seano_ca_ws/install/setup.bash
ros2 launch seano_vision run_auto_stack.launch.py
```

## Evidence

<!-- Paste output snippets, logs, or attach screenshots if relevant. -->

*

---

# Expected behavior

<!-- State what “correct” looks like so reviewers can validate quickly. -->

*

---

# Checklist (required)

* [ ] PR is focused (single feature/fix)
* [ ] `colcon build` passes locally
* [ ] No colcon artifacts committed (`build/`, `install/`, `log/`)
* [ ] No venv/cache committed (`.venv*`, `__pycache__`, `.cache`)
* [ ] Interface changes (topics/params) are documented
* [ ] No duplicate publishers introduced on critical topics (`ros2 topic info -v`)
* [ ] If actuation is affected: `/mavros/state` is `connected: true` and RC override publishes

---

# Breaking changes

<!-- If this PR changes topic names, message types, parameters, or launch behavior. -->

* [ ] None
* [ ] Yes (describe below)

## If yes, migration steps:

---

# Notes for reviewers

*

````
