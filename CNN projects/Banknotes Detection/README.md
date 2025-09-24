## 💵 Banknotes Detection with YOLO and GUI

This project implements a system for automatic banknote detection using YOLO (You Only Look Once) and convolutional neural networks, with a graphical interface built in Tkinter.
It allows dataset preparation, model training, results visualization, and performing image and real-time video classification.
The dataset was manually created on the Label Studio platform.

### 🚀 Features

* Automatic creation of data.yaml file from defined classes.
* Dataset preparation (split into train, validation, and test).
* YOLO model training with adjustable parameters:
    * Epochs
    * Batch size
    * HSV augmentation (saturation and value)
    * Learning rate (lr0) and final learning rate (lrf)
* Training results visualization:
    * Confusion matrix
    * Performance plots
    * Option to save the trained model (best.pt)
* Image classification with the trained model.
* Real-time video detection (webcam).
* User-friendly GUI (Tkinter) with two panels:
    * Original image/video
    * Classified image/video with detections

### 💶 Banknotes Classes Considered

The system was trained to recognize the following banknotes classes:

* €5 banknote  
* €10 banknote  
* €20 banknote  
* €50 banknote  
* €100 banknote  
* €200 banknote  
* €500 banknote  
* Unknown banknote (any non-euro banknote)

### 🛠️ Technologies Used

* Python 3
* Tkinter – GUI
* OpenCV – image/video processing
* Pillow (PIL) – image manipulation
* Ultralytics YOLO – object detection model
* PyYAML – YAML file handling
* Label Studio platform - create dataset and annotate

### 🖥️ How to Use

1. Open the program - the main window will appear.
2. Select one of the options in the sidebar menu:
    * Create data.yaml
    * Prepare dataset
    * Train model
    * Show results
    * Classify image
    * Classify video
3. Adjust training parameters if needed.
4. Visualize results directly in the interface.

### 📊 Example Results

Detection of different banknote denominations in an image.
Real-time banknote recognition via webcam.
Training performance reports and plots.
The results are contained in the folder "Images dos resultados".

### 👩‍💻 Author

Developed by Madalena Martins as part of the Master's in Electrical Engineering - Polytechnic Institute of Viseu. Language in which the project was developed: Portuguese.





