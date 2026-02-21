# Summary
<!-- Explain what this PR changes and why. Keep it concise. -->

## What changed?
- 

## Why?
- 

---

# Type of Change
<!-- Check ONE primary type. -->
- [ ] feat (new feature)
- [ ] fix (bug fix)
- [ ] refactor (no behavior change)
- [ ] docs (documentation only)
- [ ] test (tests only)
- [ ] chore (tooling/repo hygiene)

---

# Scope / Affected Areas
<!-- Check all that apply. -->
- [ ] Actuation / Control (teleop, mux, limiter)
- [ ] MAVROS / MAVLink bridge (RC override, FCU link)
- [ ] Simulation (SITL, Mission Planner runbook)
- [ ] Perception (camera input)
- [ ] Detection / Tracking
- [ ] Risk / Decision logic
- [ ] Launch / Config
- [ ] Documentation

---

# How to Test
<!-- Provide exact commands used. If not tested, explain why. -->

## ROS 2 Build
```bash
cd seano_ca_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install

Runtime / Demo (if applicable)
# Example: launch stack
source seano_ca_ws/install/setup.bash
ros2 launch seano_vision run_auto_stack.launch.py
Evidence
<!-- Paste output snippets, logs, or attach screenshots if relevant. -->
Expected Behavior
<!-- State what “correct” looks like so reviewers can validate quickly. -->
Checklist (Required)

 I kept the PR focused (single feature/fix)

 colcon build passes locally

 No colcon artifacts committed (build/, install/, log/)

 No virtualenv/cache committed (.venv*, __pycache__, .cache)

 Topic/interface changes are documented (README/notes)

 No duplicate publishers introduced on critical topics (checked with ros2 topic info -v where relevant)

 If this affects actuation: verified /mavros/state is connected: true and RC override publishes

Breaking Changes
<!-- If this PR changes topic names, message types, parameters, or launch behavior. -->

 None

 Yes (describe below)

If yes, describe migration steps:
Notes for Reviewers
<!-- Anything reviewers should pay extra attention to (edge cases, risks, follow-ups). -->
