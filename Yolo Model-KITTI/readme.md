# YOLO Model Trained on KITTI Dataset

This repository contains a **YOLOV5 (You Only Look Once)** model trained on the **KITTI dataset** for object detection tasks. The project includes the trained model weights, evaluation metrics, and visualizations of the training process.

## Files Included:

- **weights/**: Contains the trained model weights.
- **F1_curve.png**: F1 score curve over epochs.
- **PR_curve.png**: Precision-Recall curve.
- **P_curve.png**: Precision curve.
- **R_curve.png**: Recall curve.
- **confusion_matrix.png**: Confusion matrix for model evaluation.
- **results.png**: Training results including loss and mAP metrics.
- **train_batchX.jpg**: Visualizations of training batches.
- **val_batchX_labels.jpg**: Ground truth labels for validation batches.
- **val_batchX_pred.jpg**: Model predictions for validation batches.
- **results.csv**: Detailed training metrics for each epoch.

## Training Metrics:

The following table summarizes the training metrics over epochs:

| Epoch | Train/Box Loss | Train/Obj Loss | Train/Cls Loss | Precision | Recall | mAP 0.5 | mAP 0.5:0.95 |
|-------|----------------|----------------|----------------|-----------|--------|---------|--------------|
| 0     | 0.090416       | 0.036031       | 0.015343       | 0.78698   | 0.10426| 0.076122| 0.029643     |
| 1     | 0.073854       | 0.035428       | 0.0078914      | 0.53306   | 0.17936| 0.11724 | 0.046772     |
| ...   | ...            | ...            | ...            | ...       | ...    | ...     | ...          |
| 29    | 0.052774       | 0.032805       | 0.002777       | 0.36237   | 0.34098| 0.2408  | 0.11047      |

## Usage:

1. **Load the Trained Model**:
   - Use the weights from the `weights/` directory to load the trained YOLO model.

2. **Evaluate the Model**:
   - Use the provided scripts to evaluate the model on new data.

3. **Visualize Results**:
   - Check the `results.png`, `PR_curve.png`, and other visualization files to understand the model's performance.

## Requirements:

- Python 3.8+
- PyTorch
- OpenCV


## License:

This project is open-source and available under the MIT License.

## References:

- KITTI Dataset: [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/)
- YOLO Implementation: [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) 
