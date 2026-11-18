# License Plate Detection with YOLOv11 and EasyOCR

**Author:** Mohamad Ilham Ramadhani

## Overview
This project detects vehicles and their license plates in an image, then reads the plate text. It uses YOLO models to detect vehicles and license plates, and EasyOCR to read the characters on each plate.
The project is a Jupyter Notebook built on OpenCV, EasyOCR, PyTorch, and Ultralytics YOLO.

### Problem Statement
How can we accurately detect vehicles and their license plates in an image, and read the text on each plate?

### Objectives
- Detect vehicles in an image.
- Detect license plates within each detected vehicle region.
- Read the license plate text using OCR.

## Pipeline
```
Input Image
    ↓
Vehicle Detection (YOLOv11)
    ↓
Extract Vehicle Bounding Boxes
    ↓
License Plate Detection (YOLO, custom weights)
    ↓
Enhance License Plate Images
    ↓
Text Extraction (EasyOCR)
    ↓
Output: Annotated Image + License Plate Numbers
```

## Project Structure
```
license-plate-detection/
├── LicensePlateDetection.ipynb   # Main notebook (detection, enhancement, OCR, visualization)
├── yolo11n.pt                    # YOLOv11 nano weights (pretrained on COCO) for vehicle detection
├── license_plate_detector.pt     # YOLO weights for license plate detection
├── datasets/                     # CCTV images (local only, not included in the repository)
├── output_image.jpg              # Example output
├── .gitignore
└── README.md
```

## Dataset
The dataset has 100 vehicle images captured by CCTV cameras in the Jembatan Tiga area (Jakarta, Indonesia), under a range of conditions such as lighting, angle and distance.
The images contain motorcycles, cars, and trucks.

> **Note:** The dataset is **not included** in this repository. The `datasets/` folder is listed in `.gitignore`. To run the notebook, create a `datasets/` folder in the project root and add your own vehicle images, or point `image_path` to any image on your machine.

## Feature Processing

**Vehicle Detection**
The pretrained YOLOv11 model (`yolo11n.pt`) detects vehicles. Detection is limited to these COCO classes:

| Class ID | Class      |
|----------|------------|
| 2        | car        |
| 3        | motorcycle |
| 5        | bus        |
| 7        | truck      |

The bounding box of each detected vehicle is extracted in `xyxy` format.

**License Plate Detection**
Each vehicle region is cropped from the original image and passed to a second YOLO model (`license_plate_detector.pt`) to detect license plates. The plate coordinates are then mapped back to the coordinates of the full image.

**Image Enhancement**
- The plate region is upscaled (2× by default) using cubic interpolation.
- Histogram equalization (on the Y channel for color images) or unsharp masking can be applied to make the plate clearer.

## Modeling
- **YOLOv11** detects vehicles (motorcycles, cars, buses, and trucks) and draws a **red** bounding box around each one.
- A **YOLO license plate detector** finds the plates inside each detected vehicle and draws a **green** bounding box around each one.
- **EasyOCR** (English character set) reads the text of each detected plate. The text is drawn above the plate's bounding box.

## Model Performance
The project focuses on two capabilities:
1. Detecting vehicles reliably.
2. Detecting license plates and reading their text accurately.

No quantitative metrics, such as mAP or OCR character accuracy, have been measured yet. Evaluating against a labeled ground truth is listed under future work.

## Results and Conclusion

### Results
- The YOLO models detect vehicles and license plates well in most images.
- The main challenges are low-resolution images and difficult camera angles.
- EasyOCR performs reasonably well on clear plates but struggles with blurry or noisy ones.

### Conclusion
The system combines YOLO and EasyOCR to detect vehicles and read license plate text.
Newer YOLO versions or more advanced OCR models could improve the results further.

## Requirements
- Python 3.9+
- A CUDA-capable NVIDIA GPU (recommended; see the notes below)
- Jupyter Notebook or JupyterLab

Install the dependencies:
```bash
pip install ultralytics torch torchvision numpy opencv-python-headless matplotlib easyocr jupyter
```

> For GPU support, install the PyTorch build that matches your CUDA version. See the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

## How to Run
1. Clone or download this repository and make sure `yolo11n.pt` and `license_plate_detector.pt` are in the project root.
2. Install the dependencies listed above.
3. Prepare an input image. The dataset is not included in the repository (see [Dataset](#dataset)), so put your own image in a `datasets/` folder or anywhere else on your machine.
4. Open `LicensePlateDetection.ipynb` in Jupyter Notebook.
5. Set the `image_path` variable to the path of your image. The default value points to a file from the original dataset, which will not exist after cloning, so change it:
   ```python
   image_path = "datasets/your_image.jpg"
   ```
6. Click **Run All Cells**.
7. The notebook displays the annotated image and saves it as `output_image.jpg`.

### Notes on GPU Configuration
- The vehicle detector runs with `device=1` (the second GPU), and the plate detector runs with `device=0`. On a machine with **only one GPU**, change `device=1` to `device=0` in `detect_vehicles()`.
- To run on a machine **without a GPU**, set `device="cpu"` for both YOLO calls, set `gpu=False` in `easyocr.Reader(...)`, and remove the `.cuda()` call in `detect_vehicles()`. Expect slower inference.

## Known Limitations
- In `detect_license_plates()`, the enhanced and upscaled plate image is computed, but OCR currently runs on the original grayscale crop. Passing the enhanced image to EasyOCR may improve recognition.
- Each vehicle crop is converted from BGR to RGB before it is saved to a temporary file (`temp_vehicle.jpg`) for the plate detector. Saving it directly in BGR would keep the colors correct.
- The EasyOCR reader is initialized every time `detect_license_plates()` is called. Initializing it once and reusing it would speed up processing.

## Future Work
- Evaluate detection (mAP) and OCR accuracy against labeled ground truth.
- Feed the enhanced plate image to the OCR step and compare the results.
- Add post-processing that validates OCR output against the Indonesian license plate format (e.g., `B 1234 XYZ`).
- Support batch processing of a folder of images and video streams.
- Try newer YOLO versions or more advanced OCR models.
