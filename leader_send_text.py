#  Raspberry Pi Master for Arduino Slave
#  i2c_master_pi.py
#  Connects to Arduino via I2C
  
#  DroneBot Workshop 2019
#  https://dronebotworkshop.com

# raspberry pi IP is 10.201.22.8
# ssh throwing-a-mosfit@10.201.22.8


# rpi: port 3 yellow -> arduino SDA
# rpi: port 5 orange -> arduino SLC
# rpi: port 20 GND -> arduino GND


from smbus import SMBus

addr = 0x8 # bus address
bus = SMBus(1) # indicates /dev/ic2-1

numb = 1

print ("Enter a line of text")
while numb == 1:

    command = input(">>>>   ")
    print(f"sending command: {command}")
    bus.write_word_data(addr, 0, command)