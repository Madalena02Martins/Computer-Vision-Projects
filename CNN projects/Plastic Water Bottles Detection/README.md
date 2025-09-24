## 🍼 Plastic Water Bottles Detection with YOLO

This project implements a system for automatic detection of plastic water bottles using YOLO (You Only Look Once) and convolutional neural networks (CNNs).
It includes tools for data-augmentation, dataset preparation, model training, as well as  real-time video detection through a webcam or Intel RealSense camera.
The dataset was manually created on the Roboflow platform.

### 🚀 Features

* Data augmentation:
    * Brightness variation
    * Small random rotations
* Dataset preparation (split into train, validation, and test).
* YOLO model training with adjustable parameters:
    * Epochs
    * Batch size
* Real-time video detection:
    * Standard webcam
    * Intel RealSense camera

### ♻️ Classes Considered

The system was trained to recognize:

* Plastic water bottle 

### 🛠️ Technologies Used

* Python 3
* OpenCV – image/video processing
* Pillow (PIL) – image manipulation
* Ultralytics YOLO – object detection model
* PyYAML – YAML file handling
* Intel RealSense SDK (pyrealsense2) – real-time video capture
* Roboflow platform - create dataset and annotate

### 🛠️ How the project was created

1. The dataset images were first annotated in Roboflow and then downloaded.  
2. The script `geraimagensy.py` was executed to perform data augmentation, applying random brightness adjustments and small rotations.  
3. The dataset was then organized into training (70%), validation (15%), and testing (15%) sets using `preparadata.py`.  
4. The model was trained using the `train.py` script, where epochs and batch size were defined.
5. Finally, the trained model can be tested in real time:  
   * Using the PC webcam with `detection_camera_PC.py`.  
   * Using an Intel RealSense camera with `detection_camera_RealSense.py`.  

### 🖥️ How to Use

For real-time detection:
* Start webcam or RealSense stream.
* Exit by pressing q.

### 📊 Results

Detection of plastic water bottles.
Real-time recognition via webcam or RealSense camera.

### 👩‍💻 Author

Developed by Madalena Martins as part of a research project. 





