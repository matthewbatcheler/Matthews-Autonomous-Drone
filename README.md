# Matthew's Autonomous Drone
This project was created for the REC foundation stay-at-home drone competition. 
The system uses a Kinect depth camera to track the drone, several PID loops to determine the necessary power to apply to each control axis, then an Arduino and NRF24L01 radio to send the commands to the drone, mimicking the handset.
The video explanation of this project can be found [here](https://www.youtube.com/watch?v=F9cN-d2rOoE)
## Files
The main flight file is drone.py. colour-range.py is used to determine the HSV colour range to select each coloured marker. input_test.py is used to find the event strings from the game controller (Logitech Extreme 3D pro). drone.ggb is a geogebra project used for figuring out the trigonometry of the heading and x-y to forward-side reference frame conversion.

## Dependencies
You will require the following 3rd party libraries:
 * pySerial
 * inputs
 * imultils
 * numpy
 * openCV
 * libfreenect
 * matplotlib
 
## Usage
You will need an xbox kinect attached via usb, and an arduino + NRF24L01 running the [NRF24_CX10_PC](https://github.com/perrytsao/nrf24_cx10_pc) library configured for your drone.
You will also need a drone with two coloured markers. The colour of these markers can be determined with colour-range.py and input into the configuration section of drone.py. This section also contains other self-explanatory variables which will need to be changed.
The program makes use of a Logitech Extreme 3D pro joystick to enact manual control, as well as switch on and off the various PIDs. If you wish to change that see the read_controller() method in drone.py.

## License
This project, except where stated otherwise in the files, is covered under the GPL v3 license.
