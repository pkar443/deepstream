# pyds Installation Guide for DeepStream 6.4 + Python 3.10

This guide helps you correctly install `pyds` bindings for DeepStream 6.4 when using Python 3.10 on Ubuntu 22.04. It also explains how to verify and run your pipeline successfully without import errors.

---
## Do Not install pyds with pip
## Compatibility
- DeepStream SDK: **6.4**
- OS: **Ubuntu 22.04**
- Python: **3.10**
- Architecture: **x86_64**

---

## Download the Correct `.whl` File
You can download the precompiled `pyds` wheel file for Python 3.10 from the following link:

**[Download pyds-1.1.10-py3-none-linux_x86_64.whl](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases)**

Once downloaded, place it in your working directory `/workspace/deepstream-app/`.

---

## Step-by-Step Installation

### Step 1: Move `.whl` File to Working Directory
If the file is not already in your workspace:
```bash
mv /path/to/pyds-1.1.10-py3-none-linux_x86_64.whl /workspace/deepstream-app/
```

> If the file is already in `/workspace/deepstream-app/`, skip this step.

### Step 2: Install the `.whl` File with Python 3.10
```bash
cd /workspace/deepstream-app/
python3.10 -m pip install pyds-1.1.10-py3-none-linux_x86_64.whl
```

This ensures the package is installed using Python 3.10.

### Step 3: Verify `pyds` Installation
Run the following command to verify:
```bash
python3.10 -c "import pyds; print(dir(pyds))"
```
If successful, it will return something like:
```python
['NvDsBatchMeta', 'NvDsFrameMeta', 'gst_buffer_get_nvds_batch_meta', 'get_nvds_buf_surface', ...]
```

### Step 4: Run DeepStream Python Script
Use Python 3.10 to execute your DeepStream-based script:
```bash
python3.10 /workspace/deepstream-app/app/main_pyds.py
```

---

## 🚨 Common Errors & Fixes

### Error: `ModuleNotFoundError: No module named 'pyds'`
This happens when:
- `pyds` is not installed for the Python version you're using.
- You are using `python3` instead of `python3.10`.

Fix:
Ensure you always run with:
```bash
python3.10 your_script.py
```

---

## Best Practices
- Always match the `.whl` version to your **Python version** and **DeepStream version**.
- Use `python3.10 -m pip install` to avoid cross-version confusion.
- If developing inside a Docker container, commit the image after setup.

---

With these steps, your DeepStream + Python 3.10 environment is now ready with `pyds` support!

