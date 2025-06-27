# Object Detection

A Python-based object detection system using YOLOv5 and OpenCV to identify and track objects in images, videos, or real-time webcam streams.

## Features
- Real-Time Detection: Detects objects live using webcam.
- Image & Video Analysis: Processes local image or video files for object recognition.
- YOLOv4 Integration: Utilizes a pre-trained deep learning model for high accuracy.
- Bounding Boxes: Displays labels and confidence scores with bounding boxes.

## Getting Started

### Prerequisites
Ensure you have the following installed:
- Python 3.8 or later
- Pip (Python package installer)

### Installation

1. Clone the repository:
   bash
   git clone https://github.com/Abhinay-Pujala/Object-Detection.git
   cd Object-Detection
   

2. Install the dependencies:
   bash
   pip install -r requirements.txt
   

3. Download YOLOv4 model weights and place them in the models/ folder:
   bash
   mkdir models
   wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt -P models/
   

### Usage

Run detection on an image:
bash
python YOLO.py --source data/Your_img.jpg





Run detection using your webcam:
bash
python YOLO_WEBCAM.py --source 0


### Adding Custom Models
1. Place your .pt model file in the models/ directory.
2. Open the model_loader.py file.
3. Update the model path:
   python
   model = torch.hub.load('ultralytics/yolov4', 'custom', path='models/your_model.pt', force_reload=True)
   

## Project Structure

object-detection/
│

├── YOLO.py         # Main script to run object detection  
├── YOLO_WEBCAM.py   #  Main script to run object detection for webcam


## Technologies Used
- YOLOv4: Real-time object detection model
- PyTorch: Deep learning framework
- OpenCV: Image and video processing
- NumPy: Matrix and numerical operations

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the project.
2. Create a feature branch:
   bash
   git checkout -b feature-name
   
3. Commit your changes:
   bash
   git commit -m 'Add feature-name'
   
4. Push to the branch:
   bash
   git push origin feature-name
   
5. Open a pull request.

## Contact
For any questions or feedback, feel free to reach out:
- GitHub: [Abhinay-Pujala](https://github.com/Abhinay-Pujala)
