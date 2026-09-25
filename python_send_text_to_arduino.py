#  Raspberry Pi Master for Arduino Slave
#  Connects to Arduino via I2C
  
#  DroneBot Workshop 2019
#  https://dronebotworkshop.com

# raspberry pi IP is 10.201.22.8
# command: ssh throwing-a-mosfit@10.201.22.8

from smbus import SMBus

addr = 0x8 # bus address
bus = SMBus(1) # indicates /dev/ic2-1



print ("enter motor command in format 'MOT|DIR|SPD', type 'quit' to quit")
while True:

    command = input(">>>>   ")

    if command == "quit":
        print("quitting")
        break

    print(f"sending command to arduino: {command}")

    bus.write_byte(addr, 0x1) # switch it on


	