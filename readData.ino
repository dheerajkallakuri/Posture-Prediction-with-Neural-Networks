
#include <Arduino_LSM9DS1.h>

void setup() {
  Serial.begin(9600);
  while (!Serial);
  Serial.println("Started");

  if (!IMU.begin()) {
//    Serial.println("Failed to initialize IMU!");
    while (1);
  }
}

void loop() {
  float ax,ay,az,gx,gy,gz,mx,my,mz;
  // collecting Raw Data
  if (IMU.accelerationAvailable()) {
    IMU.readGyroscope(ax, ay, az);
  }
  if (IMU.gyroscopeAvailable()) {
    IMU.readGyroscope(gx, gy, gz);
  }
  if (IMU.magneticFieldAvailable()) {
    IMU.readMagneticField(mx, my, mz);
  }
    Serial.print(ax);
    Serial.print(",");
    Serial.print(ay);
    Serial.print(",");
    Serial.print(az);
    Serial.print(",");
    Serial.print(gx);
    Serial.print(",");
    Serial.print(gy);
    Serial.print(",");
    Serial.print(gz);
    Serial.print(",");
    Serial.print(mx);
    Serial.print(",");
    Serial.print(my);
    Serial.print(",");
    Serial.print(mz);
    Serial.print(",");
    Serial.println(5);
    delay(50);
}
