# OshaSight

**Live, always-on PPE & smoke compliance monitor** built with [YOLOv8](https://github.com/ultralytics/ultralytics) and a simple PySimpleGUI dashboard.

---

## Features

- **Six independent detectors** (Mask, Gloves, Helmet, Goggles, Hi-Viz Vest, Smoke)
- **Real-time overlays & alerts** (< 50 ms/frame, 100 FPS on GTX 1660 Ti; 31 FPS on Jetson Orin Nano)
- **Signed CSV audit log** + optional audio/visual alarms
- **Fully open-source training scripts** (VOC→YOLO converter + `train_*.py`)
- **Edge-ready deployment** (runs locally—no cloud dependency)

---
