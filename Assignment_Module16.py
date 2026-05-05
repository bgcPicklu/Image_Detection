import os
import zipfile
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

BASE_DIR = os.getcwd()

zip_file = os.path.join(BASE_DIR, "contents/Bangladesh Currency (Notes).zip")
Extracted_DIR = os.path.join(BASE_DIR, "contents/Bangladesh Currency (Notes)")
os.makedirs(Extracted_DIR, exist_ok=True)

with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(Extracted_DIR)

print('✔ Dataset has been unzipped successully')

DATA_YAML = os.path.join(Extracted_DIR, "data.yaml")

classes = ['2', '5', '10', '20', '50', '100', '200', '500', '1000']

yaml_content = f"""train: ../train/images
val: ../valid/images
test: ../test/images

nc: {len(classes)}
names: {classes}"""

with open(DATA_YAML, 'w') as f:
    f.write(yaml_content)

print('✔ data.yaml has been created successfully')

DATASET_DIR = os.path.join(BASE_DIR, "contents/Bangladesh Currency (Notes)")
TEST_IMAGES_DIR = os.path.join(DATASET_DIR, "test", "images")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

test_images = [
    f for f in os.listdir(TEST_IMAGES_DIR)
    if f.lower().endswith(IMAGE_EXTS)
]

if not test_images:
    raise FileNotFoundError("❌ No test images found in test/images directory")

TEST_IMAGE = os.path.join(TEST_IMAGES_DIR, test_images[0])

# Load image
img = Image.open(TEST_IMAGE)

# Show image
plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.axis("off")
plt.title("Sample Image")
plt.show()

train_path = DATASET_DIR + "/train/images"
val_path   = DATASET_DIR + "/valid/images"
test_path  = DATASET_DIR + "/test/images"

train_count = len(os.listdir(train_path))
val_count   = len(os.listdir(val_path))
test_count  = len(os.listdir(test_path))

total = train_count + val_count + test_count

print("Dataset Split Statistics:\n")

print("Training   :", train_count, "images", "-", (train_count/total)*100, "%")
print("Validation :", val_count, "images", "-", (val_count/total)*100, "%")
print("Test       :", test_count, "images", "-", (test_count/total)*100, "%")

print("\nTotal :", total, "images")

# Fine-tune the model using the prepared training dataset

model = YOLO('yolo26n.pt')

result = model.train(
    data=DATA_YAML,
    epochs=35,
    imgsz=192,
    batch=12,
    name="Bangladesh_Currency_Notes"
)

# ---------------------Evaluate the trained model using the test dataset------------------------ #
trained_model = YOLO("runs/detect/Bangladesh_Currency_Notes/weights/best.pt")

# Evaluate on validation/test set
metrics = trained_model.val(data="contents/Bangladesh Currency (Notes)/data.yaml", split="test")

print("Evaluation Results:")
print(metrics)
# ---------------------Evaluate the trained model using the test dataset------------------------ #

# Path to test image (same one as before)
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "contents/Bangladesh Currency (Notes)", "test", "images")
# TEST_IMAGE = os.path.join(TEST_IMAGES_DIR, os.listdir(TEST_IMAGES_DIR)[0])

# Run inference
results = trained_model.predict(
    source=TEST_IMAGES_DIR,
    save=True,          # saves output in runs/detect/
    conf=0.25,
    name="trained_inference"
)

# Display the inference image inline
OUTPUT_DIR = os.path.join(BASE_DIR, "runs", "detect", "trained_inference")
inferred_image_path = os.path.join(OUTPUT_DIR, os.listdir(OUTPUT_DIR)[0])

img = Image.open(inferred_image_path)
plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.axis("off")
plt.title("Trained Model Inference")
plt.show()

zip_file = os.path.join(BASE_DIR, "contents/Bangladesh Currency (Notes).zip")
Extracted_DIR = os.path.join(BASE_DIR, "contents/Bangladesh Currency (Notes)")
os.makedirs(Extracted_DIR, exist_ok=True)

with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(Extracted_DIR)

# Re-train or fine-tune the model to include coin detection
trained_model.train(
    data="contents/Bangladesh Currency (Notes)/data.yaml",
    epochs=35,
    imgsz=192,
    batch=12,
    name="Bangladesh_Currency_Notes2"
)

# Evaluation results and inference samples
metrics = trained_model.val(data="contents/Bangladesh Currency (Notes)/data.yaml", split="test")

# Run inference
results = trained_model.predict(
    source=TEST_IMAGES_DIR,
    save=True,          # saves output in runs/detect/
    conf=0.25,
    name="trained_inference2"
)

# Display the inference image inline
OUTPUT_DIR = os.path.join(BASE_DIR, "runs", "detect", "trained_inference2")
inferred_image_path = os.path.join(OUTPUT_DIR, os.listdir(OUTPUT_DIR)[0])

img = Image.open(inferred_image_path)
plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.axis("off")
plt.title("Trained Model Inference")
plt.show()