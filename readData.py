import serial
import time
import csv
import os

#YOUR_POSTURE = supine, prone, side, sit, unknown
#create a file of csv to store the values
try:
    os.remove("sensor_data/YOUR_POSTURE.csv")
except OSError:
    pass

#declaring headers of the csv file.
with open('sensor_data/YOUR_POSTURE.csv', mode='a') as sensor_file:
    sensor_writer = csv.writer(sensor_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    sensor_writer.writerow(["X","Y","Z","class"])

#declaring port for serial comminucation and establish connection.
com = "/dev/cu.usbmodem142201"
baud = 9600
x = serial.Serial(com, baud, timeout = 0.1)
row_c=0
# saving data in csv row by row.
while x.isOpen() == True:
    if row_c >=9000:
        break
    data = str(x.readline().decode('utf-8')).rstrip()
    if data != '':
         print(data)
         sensor_data = []
         readings = data.split(",")
         with open("sensor_data/YOUR_POSTURE.csv", mode='a') as sensor_file:
             sensor_writer = csv.writer(sensor_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
             sensor_writer.writerow([readings[0],readings[1],readings[2],readings[9]])
             sensor_writer.writerow([readings[3],readings[4],readings[5],readings[9]])
             sensor_writer.writerow([readings[6],readings[7],readings[8],readings[9]])
             row_c+=3

