---
title: BUFFER-X Hub Helper
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
suggested_hardware: cpu-basic
---

# BUFFER-X Hub Helper

This Space is a lightweight companion for BUFFER-X. It checks that the model
repository exposes the expected pretrained snapshot layout and generates install
and download commands for users.

The full BUFFER-X inference path requires CUDA extensions, so this helper is kept
CPU-only and reliable. For an interactive registration demo, duplicate this Space
and extend the Dockerfile with the CUDA installation path in the GitHub repository.
