from adafruit_pca9685 import PCA9685
from board import SCL, SDA
import busio
import time

i2c = busio.I2C(SCL, SDA)
pca = PCA9685(i2c)
pca.frequency = 50  # for servos

channel = pca.channels[0]
while True:
    for pulse in [1000, 1500, 2000]:  # microseconds
        channel.duty_cycle = int(pulse / 20000 * 65535)
        print("Duty cycle:", channel.duty_cycle)
        time.sleep(2)
