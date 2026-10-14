# throw-a-mosfit-pi


Using This Codebase:
This code has two main purposes:
1. interpret the pi's vision into decisions (this should happen as quickly as possible)
2. show the vision and decisions to the user through a GUI so we can understand what is happening


This is accomplished by decoupling the video / photo / live feed and the results, so it is easier to debug and understand what is happening.


Wiring:
RPi: with ports at the bottom, 
Ground: right column 10th down (20) to mega GND
Yellow SDA left 2nd (3) to mega SDA
Orange SCL left 3rd (5) to mega SDA

Connect to pi from computer:
ssh throwing-a-mosfit@10.201.22.8
password is TAM

