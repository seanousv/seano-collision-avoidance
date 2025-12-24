# SEANO Collision Avoidance (Camera-Based)

Modul **Collision Avoidance berbasis kamera (vision-only)** untuk USV SEANO.  
Target runtime: **ROS 2 Humble**, komputasi Jetson (Orin Nano), pipeline CV/AI.

> **Scope repo ini:** khusus **Collision Avoidance (CA)**.  
> Modul lain (dashboard, komunikasi, data logging, path planning, dll) dikerjakan di repo terpisah.

---

## Repo Structure

seano_ca_ws/
src/
seano_vision/ # ROS 2 package (camera bridge)
seano_vision/
camera_node.py
package.xml
setup.py
setup.cfg


---

## Requirements

- Ubuntu 22.04
- ROS 2 Humble
- OpenCV + cv_bridge

Install dependencies:
```bash
sudo apt update
sudo apt install -y python3-opencv ros-humble-cv-bridge
sudo apt install -y ros-humble-rqt-image-view   # optional viewer

---

## Quick Start (Build & Run)

### 1) Clone repo
```bash
git clone https://github.com/seanousv/seano-collision-avoidance.git
cd seano-collision-avoidance
