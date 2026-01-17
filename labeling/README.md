# Labeling Workflow (Signature/Stamp)

## Option A: Label Studio (recommended)
Use a separate environment to avoid NumPy conflicts with Ultralytics.

### 1) Create a new environment
```
python -m venv /Users/ryanfernandes/convolve/labeling/.venv
/Users/ryanfernandes/convolve/labeling/.venv/bin/python -m pip install label-studio
```

### 2) Start Label Studio
```
/Users/ryanfernandes/convolve/labeling/.venv/bin/label-studio start
```

### 3) Create Project
- Import images from [yolo_data/images/train](yolo_data/images/train)
- Use labeling config in [labeling/label_studio_config.xml](labeling/label_studio_config.xml)
- Draw boxes for `signature` and `stamp`

### 4) Export and Convert
Export in JSON format, then convert to YOLO labels:
```
/Users/ryanfernandes/convolve/.venv/bin/python \
  labeling/convert_label_studio_to_yolo.py \
  --ls_json /path/to/label_studio_export.json \
  --images_dir /Users/ryanfernandes/convolve/yolo_data/images/train \
  --labels_dir /Users/ryanfernandes/convolve/yolo_data/labels/train
```
Repeat for validation images.

## Training After Labels
Once labels are present, run YOLO training using the dataset config in [yolo_data/dataset.yaml](yolo_data/dataset.yaml).
