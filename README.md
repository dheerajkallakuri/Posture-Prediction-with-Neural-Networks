# Posture Classification System

## Overview

This project aims to develop a sensor-agnostic posture classification system that uses an IMU sensor unit embedded in an Arduino board. The system will gather sensor data, including 3-axis accelerometer, 3-axis gyroscope, and 3-axis magnetometer, and implement a machine learning algorithm for real-time posture detection. The final model will be deployed on the microcontroller for live predictions and communicate the results to a base station (laptop or smartphone).

## Project Phases

### Phase 1: IMU Sensor Data Collection
Run `readData.ino` code to read IMU sensor data and store signal readings on your computer.

### Phase 2: Data Collection
Run `readData.py` Collect data for different postures (supine, prone, side, sitting, and unknown). Ensure to gather data for various sensor orientations to ensure robustness.

### Phase 3: Dataset Construction
Construct `data.csv` with all 3 sensors data for all 5 postures and split it into training, validation, and test sets. Train the model with only 3 input channels (x, y, and z).

### Phase 4: Neural Network Architecture
Run `Relu_datamodel.py` a custom neural network architecture and train your model offline.
Obtain `model_pred.tflite` and `model_pred_quant.tflite`

### Phase 5: Model Performance Assessment
Assess the performance of your model. Make adjustments to the architecture and dataset to prevent overfitting or underfitting.

### Phase 6: Model Testing
Test your model on the test dataset.

### Phase 8: Deployment and rediction Interface
Convert `model_pred.tflite` or `model_pred_quant.tflite` into a `model.cc` source file for Arduino BLE Sense.

## Robustness Considerations
- Ensure the model is insensitive to sensor orientation changes.
- Collect data representing the same posture in different orientations and label them consistently.
- Discuss assumptions about sensor positioning, operating points, and corner cases in the report.

## Results
<div>
  <img width="585" alt="prediction" src="https://github.com/dheerajkallakuri/Posture-Prediction-with-Neural-Networks/assets/23552796/1a9301c5-76ca-4666-bb2d-7875158671e7">
</div>

- The following has a prediction value of about 81% because of the gyroscope sensor.
- The gyroscope measured the angular velocity of the system and when we were collecting data we kept it at rest so the angular velocity values were kind of gibberish.
- The value of the magnetometer affects the magnetic field of the thighs around it. So we made sure that during data collection and inference, we kept the same setup.

## Video Demonstration

For a visual demonstration of this project, please refer to the video linked below:

[Project Video Demonstration](https://youtu.be/GaMZuCQiNEI)

[![Project Video Demonstration](https://img.youtube.com/vi/GaMZuCQiNEI/0.jpg)](https://www.youtube.com/watch?v=GaMZuCQiNEI)



