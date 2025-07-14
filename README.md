---
title: InstantID
emoji: 😻
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 4.40.0
app_file: app.py
pinned: false
license: apache-2.0
disable_embedding: true
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


## Instructions:

Create a virtual environment with Python 3.10

Activate the virtual environment
```sh
pip install -r requirements.txt
```
```sh
python download_models.py
```
```sh
python app.py
```
> And in app.py, you can try changing `face_file`, `pose_file`, and `prompt` if you want to test with different inputs.
