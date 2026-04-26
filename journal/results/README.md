# Journal Dynamic Obstacle Test Results

This directory contains summarized experimental results for the paper:

Mission-Aware Collision Avoidance System for a Differential-Thrust Unmanned Surface Vehicle Using AI-Based Obstacle Detection.

## Included files

- dynamic_obstacle_results.csv: summary of 10 repeated dynamic-obstacle trials.
- ../tools/analyze_dynamic_trial.py: parser used to extract trial-level metrics from event logs.

## Test scope

The dataset represents repeated evaluation using a dynamic-obstacle video scenario. It validates repeatability of the mission-aware stop-observe-escape behavior, including avoidance activation, maneuver generation, differential-thrust command output, and mission recovery.

Raw ROS logs, event frames, video files, bag files, and MAVLink telemetry logs are intentionally not committed to keep the repository lightweight.
