/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <TensorFlowLite.h>
#include <Arduino_LSM9DS1.h>

#include "main_functions.h"
#include "model.h"
#include "output_handler.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

constexpr int kTensorArenaSize = 2000;
// Keep aligned to 16 bytes for CMSIS
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}  // namespace
int sensorChoice=0;
// The name of this function is important for Arduino compatibility.
void setup() {
  Serial.begin(115200);
  while (!Serial);

  // Initialize the LSM9DS1 sensor
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  // Prompt the user to select a sensor
 Serial.println("Select a sensor to read: Option{1,2,3}");
  Serial.println("Accelerometer >> 1");
  Serial.println("Gyroscope >> 2");
  Serial.println("Magnetometer >> 3");
  while (!Serial.available());

  sensorChoice = Serial.parseInt();
  tflite::InitializeTarget();

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Keep track of how many inferences we have performed.
  inference_count = 0;
}

// The name of this function is important for Arduino compatibility.
void loop() {
  // Calculate an x value to feed into the model. We compare the current
  // inference_count to the number of inferences per cycle to determine
  // our position within the range of possible x values the model was
  // trained on, and use this to calculate a value.
  float x,y,z;
  if(sensorChoice == 1 && IMU.accelerationAvailable()) {
    IMU.readAcceleration(x, y, z);
  }
  else if(sensorChoice == 2 && IMU.gyroscopeAvailable()) {
    IMU.readGyroscope(x, y, z);
  }
  else if(sensorChoice == 3 && IMU.magneticFieldAvailable()) {
    IMU.readMagneticField(x, y, z);
  }
  // float position = static_cast<float>(inference_count) /
  //                  static_cast<float>(kInferencesPerCycle);
  // float x 

  // Place the quantized input in the model's input tensor
  Serial.println("x , y ,z");
  Serial.println(x);
  Serial.println(y);
  Serial.println(z);
  input->data.f[0] = x;
  input->data.f[1] = y;
  input->data.f[2] = z;
  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed on x: %f\n", static_cast<double>(x));
    return;
  }
 float y_quantizedd[5];
  float a[5];
  float cl,ot;
  cl=0;
  Serial.println("output");
  for (int i = 0; i < 5; i++) {
      a[i] = output->data.f[i];
      Serial.println(a[i]);
  }
  for (int i = 0; i < 5; i++) {
      ot=a[i];
      if(ot>cl){
        cl=round(ot);
        }
  }
  Serial.println("pred:");

  Serial.println(cl);

  switch (int(cl))
  {
    case 1:
      Serial.println("Posture: Supine");
      break;
    case 2:
      Serial.println("Posture: Prone");
      break;
    case 3:
      Serial.println("Posture: Side");
      break;
    case 4:
      Serial.println("Posture: Sitting");
      break;

    default:
      Serial.println("Invalid Posture prediction");
      break;
  }

  delay(1000);
  
}
