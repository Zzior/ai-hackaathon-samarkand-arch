from pathlib import Path


project_dir = Path(__file__).parent.parent.parent.parent

train_img_info_file = project_dir / "media/tc_dataset/Train.txt"
val_img_info_file = project_dir / "media/tc_dataset/Validation.txt"

train_img_path = project_dir / "media/tc_dataset/images/Train"
train_labels_path = project_dir / "media/tc_dataset/labels/Train"

val_img_path = project_dir / "media/tc_dataset/images/Validation"
val_labels_path = project_dir / "media/tc_dataset/labels/Validation"
for pth in (train_img_path, train_labels_path, val_img_path, val_labels_path):
    pth.mkdir(parents=True, exist_ok=True)

train_pic: list[Path] = [file for file in train_img_path.iterdir() if file.is_file()]
val_pic: list[Path] = [file for file in val_img_path.iterdir() if file.is_file()]


# Init and move val labels ====================================================
val_img_info_text = ""
for val_path in val_pic:
    pth = train_labels_path / val_path.name
    pth = pth.with_suffix(".txt")
    val_img_info_text += f"data/images/Validation/{val_path.name}\n"

    if pth.is_file():
        pth.rename(val_labels_path / pth.name)  # move labels

if val_img_info_text:
    with open(val_img_info_file, "w") as f:
        f.write(val_img_info_text)

# Init train labels ===========================================================
train_img_info_text = ""
train_labels_files = set()
for train_path in train_pic:
    pth = train_labels_path / train_path.name
    pth = pth.with_suffix(".txt")
    train_labels_files.add(pth)

    train_img_info_text += f"data/images/Train/{train_path.name}\n"

with open(train_img_info_file, "w") as f:
    f.write(train_img_info_text)


# Delete unused labels ========================================================
for label_file in train_labels_path.iterdir():
    if label_file not in train_labels_files:
        label_file.unlink()


# Create data.yaml for CVAT
data_yaml = """Train: Train.txt
Validation: Validation.txt
names:
  0: person
  1: car
  2: traffic_light
  3: crosswalk
path: .
"""

# Create data.yaml for YOLO
data_yolo_yaml = f"""names:
- person
- car
- traffic_light
- crosswalk
nc: 4
path: {project_dir}/media/tc_dataset
train: images/Train
val: images/Validation
test: images/Validation
"""

with open(project_dir / "media/tc_dataset/data.yaml", "w") as f:
    f.write(data_yaml)

with open(project_dir / "media/tc_dataset/data_yolo.yaml", "w") as f:
    f.write(data_yolo_yaml)
