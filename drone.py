'''
Using an Xbox Kinect and arduino with the nrf24_cx10_pc library, flies a quadcopter autonomously
Copyright (C) 2020 Matthew Batcheler

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''
import logging
import threading
import math
import time


import serial
from inputs import get_gamepad

import sys

from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import freenect

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation


loc = '/dev/ttyACM0'

#### ---- basic config

origin_x = 183 #origin of drone zone (the whole image is not used to track by default.)
origin_y = 0 #see above. remember that for images y is 0 at the top and at a maximum at the bottom of the image, the opposite of a cartesian plane

width = 580 - origin_x #width of the drone zone 
height = 447 #see above

#[ [x,y,z], [x2, y2, z2], ...]
#sequence = [[127, 250, 60], [253, 246, 60], [253, 123, 60], [127, 123, 60]]
sequence = [[100, 380, 60], [297, 380, 60], [199, 240, 60], [100, 100, 60], [297, 100, 60], [100, 380, 60]] #this is the sequence of waypoints the drone follows

z_groundheight = 929 #the z distance that the ground level is at.

image_display_size = 900 #how wide to blow up the image for display.

#colour definitions for tracking - to track, the code locks onto two different colour markers. The range of colour values in HSV format for each is defined
blueLower = (79, 82, 0) #super sensitive blue
blueUpper = (131, 251, 255)

#orangeLower = (0, 70, 195) #super sensitive orange - good w/ artificial light
#orangeUpper = (80, 255, 255)

#orangeLower = (0, 106, 208) #super sensitive orange - good w/ daylight
#orangeUpper = (90, 255, 255)


orangeLower = (25, 0, 110)  #I changed this to green at the last moment, but its just the second colour marker whatever that may be
orangeUpper = (86, 233, 255)


#### ---- Advanced config

depth_boxwidth = 100 #when figuring out the drone's height, a box is taken around the expected location, and the highest point is found
                     #this is because there is some error around the drone's actual position on the depth image due to lens offset
depth_boxheight = 100



###########-------------------INTERNAL VARIABLES, DO NOT CHANGE

#### ---- tracking variables

x = 0 #left right 
y = 0 #fwd-back
y_raw = 0 #dumb stupid inverse y
z = 0 # up down
h = 0 #heading degrees from north

old_x = 0 #last trusted values, since the tracker might lose the lock for a bit
old_y = 0
old_z = 0
old_y_raw = 0 


#### ---- internal variables for controller and pids

#starting (passive) control values
throttle = 1000 #1000 (off) - 2000 (full)
roll = 1500 #1000 (hard left) - 2000 (hard right)
pitch = 1500 #1000-2000 (i think 2000 is forward, 1000 is back?)
yaw = 1500 #1000 (hard left) - 2000 (hard right)

#pid targets

z_trg = sequence[0][2] #altitude in arbitrary units
h_trg = 0 #heading
x_trg = sequence[0][0]
y_trg = sequence[0][1] #pixels from adjusted y (so y=0 is bottom rather than dumb top)
current_step = 0


#used to track which pids are on or off
ena_pid_vertical = False #do vertical pid
ena_pid_h = False
ena_pid_xy = False


xy_reset = False #reset xy pid. used to reset when target is changed to avoid derivative spike

next_waypoint_timer = 0


#logging for graph
throttle_log = []
throttle_log_x = []

z_log = []
z_log_x = []

p_log = []
p_log_x = []

i_log = []

d_log = []
d_log_x = []

error_log = []
error_log_x = []

target_log = []

output_log = []

pid_was_on = False

launch_land = 0

def crop_to_zone(frame): #crops a frame to the drone zone
    return frame[origin_y:height+origin_y, origin_x:width+origin_x]

def getImage():	 #get image from kinect and swap coloour channels to BGR (for opencv)
    image, timestamp = freenect.sync_get_video()
    red = image[:,:,2].copy()
    blue = image[:,:,0].copy()
    image[:,:,0] = red
    image[:,:,2] = blue
    return crop_to_zone(image)


def getDepthVal(x,y): #get depth at a specific point

    if(x > width or x < 0 or y > height or y < 0): #check the request is inside the drone zone
        print("out of bounds")
        return 2047

    box_half_width = int((depth_boxwidth)/2)
    box_half_height = int((depth_boxheight)/2)

    depth, timestamp = freenect.sync_get_depth() #get image
    depth = crop_to_zone(depth) #crop to drone zone

    #since the depth and rgb images don't align exactly (bcos cameras are slightly offset), define a region around the approx drone location
    drone_area = depth[y-box_half_height:y+box_half_height, x-box_half_width:x+box_half_width]

    val = 2047 # 2047 = failed :(
    if(drone_area.shape[0] > 7 and drone_area.shape[1] > 7): #make sure the drone zone is greater than 7x7 for the gauss blur
        drone_area = cv2.GaussianBlur(drone_area, (7, 7), 0) #blur the pixel values a bit
        
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(drone_area) #find the extrema of the image - furtherst and closest

        np.clip(drone_area, 0, 2**10 - 1, drone_area) #do some processing in case i want to display the image
        drone_area >>= 2
        drone_area = drone_area.astype(np.uint8)

        #cv2.imshow("depth", drone_area) #display the depth image if wanted

        val = z_groundheight - minVal #take the minimum distance (closest to the camera, so must be the drone) and make it into a sensible height value
    return val

def process_throttle(t): #scale value from controller to 1000-2000 type
    t = 255-t
    return int((t * 1000/255 + 1000))

def process_yaw(y): #scale value from controller to 1000-2000 type
    return int((y * 1000/255 + 1000))

def process_pitch(p): #scale value from controller to 1000-2000 type
    p = p * 1000/1023 #0-1000
    p = 1000-p #invert
    return int(p+1000)

def process_roll(r): #scale value from controller to 1000-2000 type
    r = r * 1000/1023 #0-1000

    return int(r+1000)

def read_lines_in(name): # read and print lines from the arduino. This runs in a thread. the arduino is the board handling the actual radio transmission
    global arduino, throttle, roll, pitch, yaw
    while(1):
        data = arduino.readline()
        if data:
            #String responses from Arduino Uno are prefaced with [arduino]
            logging.info("[arduino]: "+str(data))
            time.sleep(0.5)


def time_now(): #gets the time (seconds w decimals) since the program started
    global start_time
    return time.time() - start_time

def graph(name): #shows a graph of some pid values when pid is turned off. runs in thread. kinda a mess, needs making nicer
    global pid_was_on, ena_pid_xy, ena_pid_vertical
    while(True):
        if(not ena_pid_vertical and pid_was_on):
            fig, ax = plt.subplots()
            print(error_log)
            print(p_log)
            print(output_log)
            ax.plot(throttle_log_x, z_log, label='z height')
            ax.plot(throttle_log_x, throttle_log, label='throttle')
            ax.plot(throttle_log_x, error_log, label='error')
            ax.plot(throttle_log_x, target_log, label='target')
            ax.plot(throttle_log_x, p_log, label='p')
            ax.plot(throttle_log_x, i_log, label='i')
            ax.plot(throttle_log_x, d_log, label='d')
            ax.plot(throttle_log_x, output_log, label='output b4 hover constant')
            
            ax.legend()
            #ax.plot(t2, s2)

            ax.set(xlabel='time (s)', ylabel='powah',
                title='drone')
            ax.grid()

            fig.savefig("test.png")
            plt.show()
            pid_was_on = False
        else:
            if(ena_pid_vertical):
                pid_was_on = True
        time.sleep(0.1)

def read_controller(name): #reads the vals and buttons from the controller, setting them as global values (goes to drone) or ena/disabling features like pid
    global arduino, throttle, roll, pitch, yaw, ena_pid_vertical, ena_pid_h, ena_pid_xy, stopped, launch_land
    stopped = False #is execution halted because of crash avoidance (trigger button)
    while not stopped:
        events = get_gamepad() #get the events array from the joystick

        for event in events: #loop thru it

            if(event.code == "ABS_Y" and not ena_pid_xy): #pitch input, check pid is off

                pitch = process_pitch(int(event.state)) #scale it then set as global

            elif(event.code == "ABS_X" and not ena_pid_xy):

                roll = process_roll(int(event.state))

            elif(event.code == "ABS_RZ" and not ena_pid_h):

                yaw = process_yaw(int(event.state))

            elif(event.code == "ABS_THROTTLE" and not ena_pid_vertical):

                throttle = process_throttle(int(event.state))

            elif(event.code == "BTN_TRIGGER"): #this is the emergency stop button. in theory, cuts throttle, restarts arduino to drop connection, then stops prog execution
                throttle = 1000
                ena_pid_vertical = False
                stopped = True
                time.sleep(0.2)
                arduino.close()
                # re-open the serial port which will also reset the Arduino Uno and
                # this forces the quadcopter to power off when the radio loses conection. 
                arduino=serial.Serial(loc, 115200, timeout=.01)
                arduino.close()
                time.sleep(1)
                sys.exit("emergency stop called")

            elif(event.code == "BTN_THUMB" and event.state == 1): #toggle vertical axis pid
                ena_pid_vertical = not ena_pid_vertical

            elif(event.code == "BTN_BASE" and event.state == 1): #toggle heading pid
                ena_pid_h = not ena_pid_h

            elif(event.code == "BTN_BASE2" and event.state == 1): #toggle xy pid
                ena_pid_xy = not ena_pid_xy
            elif(event.code == "BTN_BASE6"): #prints out current throttle value
                print("t: %i, p: %i, r: %i, y: %i"%(throttle, pitch, roll, yaw))
            elif(event.code == "BTN_BASE4" and event.state == 1):
                sequencer_step(1)
            elif(event.code == "BTN_BASE3" and event.state == 1):
                sequencer_step(-1)
            elif(event.code == "BTN_BASE5" and event.state == 1): #launch the drone
                if(launch_land == 0):
                    launch_land = -1
                launch_land = -math.copysign(1, launch_land)
                print("launch_land = %i"%(launch_land))


def update_drone(name): #sends current global variables (throttle, pitch, roll, yaw) to arduino to be transmitted
    time.sleep(5) #wait a bit for the connection to sort itself out
    logging.info("Started arduino control")
    global throttle, roll, pitch, yaw
    while(1):

        #print(throttle, roll, pitch, yaw)
        command="%i,%i,%i,%i"% (throttle, roll, pitch, yaw)
        arduino.write(str.encode(command + "\n")) #write command
        time.sleep(0.1)


def find_colour_marker(frame,lower, upper): #blurred frame in HSV, lower HSV bound, upper HSV bound
    mask = cv2.inRange(frame, lower, upper) #create a mask of the specific colours
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours) #find contours
    center = None
    if len(contours) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(contours, key=cv2.contourArea) #find the biggest contour
        ((x, y), radius) = cv2.minEnclosingCircle(c) # find the circle around this contours
        M = cv2.moments(c) #moments the circle - to calc center
        if(M["m00"] > 0):
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) #calculate the center
        
        # only proceed if the radius meets a minimum size
        if radius > 2:#   
            return center
        else:
            return False

def calc_heading(orange, blue, center): #calculates the heading angle from the two points (harder than it sounds :/)
    dx1 = center[0] - orange[0]  #disance between marker and center
    dy1 = orange[1] - center[1] #flipped because of reversed y axis (not using adjusted y)

    dx2 = blue[0] - center[0] #see above except for the blue marker
    dy2 = center[1] - blue[1] #flipped because of reversed y axis

    dx = (dx1+dx2)/2 #average x distance to the center
    dy = (dy1+dy2)/2 #average y distance to the center

    h=0

    if(dy != 0): #avoid dividing by 0
        if(dy > 0): #top two quadrants and horz axis

            h = math.degrees(math.atan(dx/dy)) + 225
        if(dy < 0): #bottom quadrants

            h = math.degrees(math.atan(dx/dy)) + 405
    else:
        if(dx < 0): # right of origin
            h = 135
        elif(dx > 0): #left of origin
            h = 315
    h-= 45
    if(h >= 360): #limit to 0<h<360
       h -= 360

    return h


def track_drone(name): #gets the latest image and find the drone's position and heading. runs in a thread
    global x, y, z, old_x, old_y, old_z, h, y_raw, old_y_raw, x_trg, y_trg, height
    bad_frame = 0
    while True:

        # grab the current frame
        frame = getImage()
        
        if frame is None: 
            print("frame error!")
            break

        # resize the frame, blur it, and convert it to the HSV
        # color space
        blurred = frame#cv2.GaussianBlur(frame, (11, 11), 0) 
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV) #convert the image to HSV

       

        blue = find_colour_marker(hsv, blueLower, blueUpper) #find blue colour marker
        if(blue != False): #draw it onto the frame
            cv2.circle(frame, blue, 3, (255, 0, 0), -1)
        #else:
            #print("failed to find blue")

        orange = find_colour_marker(hsv, orangeLower, orangeUpper) #ditto for orange
        if(orange != False):
            cv2.circle(frame, orange, 3, (0, 51, 255), -1)
        #else:
            #print("failed to find orange")

        if(blue != False and orange != False and blue != None and orange != None): #check it found the tracking markers and didnt break

            x = int((orange[0] + blue[0])/2) #find center of the drone (halfway between markers)
            y_raw = int((orange[1] + blue[1])/2)

            z = getDepthVal(x,y_raw) #get the height

            y = height - y_raw #reverse y axis to stop it doing my head in (images y axis counts downwards positively aaaaaaaaaaaaaaaaaaaaaaa)

            

            cv2.circle(frame, (x,y_raw), 3, (0, 0, 0), -1) #draw circle for current position
            cv2.circle(frame, (x_trg,height - y_trg), 3, (0, 0, 0), -1) #draw circle for target position

            h = calc_heading(orange, blue, (x,y_raw)) #calculate the current heading

            #draw an arrow in the direction its pointing
            cv2.arrowedLine(frame, (x,y_raw), (int(70*math.cos(math.radians(90-h)) + x), -int(70*math.sin(math.radians(90-h))) + y_raw), (0,0,255), 2)

            #resize the frame for the display
            frame = imutils.resize(frame, width=image_display_size)

            #draw debug text
            cv2.putText( frame, "height: " + str(z), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1,  (209, 80, 0, 255), 3) 
            
            cv2.putText( frame, "heading: " + str(h), (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1,  (209, 80, 0, 255), 3) 

            cv2.putText( frame, "x: %i, y: %i, y_raw: %i"%(int(x),int(y),int(y_raw)), (20,120), cv2.FONT_HERSHEY_SIMPLEX, 1,  (209, 80, 0, 255), 3) 
           

            old_x = x #set the old values in case the next tracking frame fails
            old_y = y
            old_z = z
            old_y_raw = y_raw
            bad_frame = 0 #reset bad frame counter
            
        else:
            x = old_x #if it failed, set position as old position
            y = old_y
            z = old_z
            y_raw = old_y_raw
            bad_frame += 1 #increase bad frame counter
            frame = imutils.resize(frame, width=image_display_size) #resize frame for display
        #if(bad_frame > 10):
            #print("too many bad frames!")
        
        cv2.imshow("Frame", frame) #show frame
        # show the frame to the screen
        
        
        cv2.waitKey(10)



def do_pid_vertical(name):
    global z, z_trg, throttle, ena_pid_vertical, z_kP, p, d, stopped, launch_land
    throttle_cap = 400
    old_error_lst = [0, 0, 0] #preps the old error list
    while not stopped:
        z_kP = 5 #constants - should probably go in the top of the file
        z_kD = 25
        z_kI = 0.5
        integral=0
        if(ena_pid_vertical and z != 2047): #2047 means it lost track so stop going upwards
            
            error = z_trg - z

            if(launch_land > 0): #drone is launching so keep gradual upwards drection to avoid oscillations
                error = 20
                #print("launching")
                
                if(abs(z_trg - z) < 15):
                    print("launch off")
                    launch_land = 0
            if(launch_land < 0): #landing
                z_trg = 0
                error = z_trg - z
            
            old_error = sum(old_error_lst)/ len(old_error_lst) #avg dz/dt of the last few loops
            p = z_kP * error
            #if(error < 0):
            #    d = (error-old_error) * z_kD * 2
            #else:
            d = (error-old_error) * z_kD
            if(abs(d) > 200):
                d = 0
            

            
            if(abs(error) < 10):
                integral -= error
            else:
                integral += error
            i = integral * z_kI
            output = p + d + i

            if(abs(error) > 200):
                print("reset itnegral")
                integral = 0
                i = 0
            
            if(abs(output) > throttle_cap): #limit to -500 : 500
                output = math.copysign(throttle_cap, output)

            throttle = output + 1225
            if(throttle < 1000):
                throttle = 1000
            if(throttle > 2000):
                throttle = 2000


            if(launch_land == -1 and error < 10):
                throttle = 1000
                ena_pid_vertical = False

            #print("current: %f, target: %f, error: %f, p: %f, i: %f, d: %f, throttle %f"%(z, z_trg, error, p, i, d, throttle))
            
            throttle_log.append(throttle)
            throttle_log_x.append(time_now())

            error_log.append(error)
           # error_log_x.append(time_now())

            p_log.append(p)
           # p_log_x.append(time_now())

            d_log.append(d)
           # d_log_x.append(time_now)

            z_log.append(z)
            i_log.append(i)
            #z_log_x.append(time_now())

            target_log.append(z_trg)

            output_log.append(output)

            if(len(old_error_lst) > 15):
                #print("removing:")
                old_error_lst.pop(0)
            old_error_lst.append(error)
            #print(old_error_lst)
            
        elif(ena_pid_vertical and z == 2047):
           throttle = 1000
        time.sleep(0.01)

def do_pid_heading(name):
    global h, h_trg, yaw, ena_pid_h
    h_kP = 2
    h_kI = 0
    h_kD = 0

    while True:
        if(ena_pid_h):
            error = h - h_trg #anticlockwise is positive
            if(abs(error) > 180): #would be easier to go the other way, across the 0 point
                 #from large positive bearing to small negative bearing
                error = h - (h_trg - 360)

            p = error * h_kP

            power = error * h_kP

            if(abs(power) > 500):
                math.copysign(500, power)
            yaw = -power + 1500
            #print("current: %f, target: %f, error: %f, p: %f, power %f"%(h, h_trg, error, error * h_kP, power))#, i, d, throttle)), i: %f, d: %f, throttle %f
        time.sleep(0.01)
            
def do_pid_xy(name):
    global x, y, x_trg, y_trg, h, roll, pitch, throttle_log_x, error_log, p_log, d_log, xy_reset, next_waypoint_timer
    old_error_fwd = 0
    old_error_side = 0
    old_error_lst_fwd = [0, 0, 0]
    old_error_lst_side = [0, 0, 0]
    fwd_integral = 0
    side_integral = 0
    time_rn = 1.5
    next_waypoint_timer_start = 0
    xy_start_time = 0
    while True:
        
        xy_kP = 0.45
        fwd_kP = 0.45
        side_kP = 0.45
        xy_kI = 0.015
        xy_kD = 2#.6
        if(ena_pid_xy):

            
            x_error = x_trg - x
            y_error = y_trg - y

            L = math.sqrt((x_error)**2 + (y_error)**2)

            if(x_error != 0): #avoid dividing by 0
                gamma = math.degrees(math.atan(y_error / x_error)) #see drawings in repo for explanation of angles
            else:
                print("div by 0, so gamma is 90/-90")
                gamma = math.copysign(90, y_error)
            if(x_error >=0): #first and fouth quadrants, and the centerline - watch out for div by 0!
                #print("right side")
                delta = 90 - h - gamma
            else:
                #print("left side")
                delta = 270 - h - gamma

            fwd_error = L * math.cos(math.radians(delta))
            side_error = L * math.sin(math.radians(delta))

            old_time_rn = time_rn
            time_rn = time_now()
            if(time_rn - old_time_rn > 1 or xy_reset): #is the time between the last loops time_rn and now greater than 1 sec? if so, pid has been turned off so reset 
                print("reset xy pid")
                xy_start_time = time_rn
                fwd_integral = 0
                side_integral = 0
                old_error_lst_fwd = [fwd_error] * 7 #initialise the derivative moving mean array to avoid an initial spike
                old_error_lst_side = [side_error] * 7

                xy_reset = False
            old_error_fwd = sum(old_error_lst_fwd)/ len(old_error_lst_fwd) #average old error from the past few loops, smooths out otherwise very jerky erroneous measurements. should probably use kalman filters lmao
            old_error_side = sum(old_error_lst_side)/ len(old_error_lst_side)

            if(L < 50):
                #print("inc timeer: %f, %f"%(next_waypoint_timer, next_waypoint_timer_start))
                if(next_waypoint_timer == 0):
                    next_waypoint_timer_start = time_rn
                next_waypoint_timer = time_rn
            else:
                print("reset")
                next_waypoint_timer = 0
                next_waypoint_timer_start = 0

            if(next_waypoint_timer - next_waypoint_timer_start > 2):
                print("next waypoint!")
                sequencer_step(1)
            

            if(abs(fwd_error) < 5): #only do integral in certain range
                fwd_integral -= -fwd_error
            else:
                fwd_integral += fwd_error #integral terms before multiplying by kI
            
            if(abs(side_error) < 5):
                side_integral -= -side_error
            else:
                side_integral += side_error #integral terms before multiplying by kI
                

            if(abs(fwd_error) > 85):
                fwd_integral = 0
            if(abs(side_error) > 85):
                side_integral = 0
                

            fwd_d = (fwd_error - old_error_fwd) * xy_kD #calcualte final derivative term
            side_d = (side_error - old_error_side) * xy_kD

            if(time_rn - xy_start_time < 0.3): #disable derivative for the first sec to avoid jumping
                print("d off")
                fwd_d = 0
                side_d = 0
            
            fwd_i = fwd_integral  * xy_kI #calcualte final integral term
            side_i = side_integral * xy_kI

            #      proportional term --  derivative -      ---Integral term
            #                         V             V     V
            output_fwd = (fwd_kP * fwd_error)  + fwd_d + fwd_i
            output_side= (side_kP * side_error) + side_d+ side_i

            if(abs(output_fwd) > 500): #limit range of outputs to the avaliable control range (1000-2000 so +/- 500)
                output_fwd = math.copysign(500, output_fwd)
            if(abs(output_side) > 500):
                output_side = math.copysign(500, output_side)

            roll = 1500 + output_side #add the static value (1500), halfway between full back and full forward
            pitch = 1500 + output_fwd

            ##LOGGING STUFF START
            '''throttle_log_x.append(time_now())
            error_log.append(fwd_error)

            i_log.append(fwd_i)

            d_log.append(fwd_d)

            p_log.append(fwd_kP * fwd_error)

            output_log.append(output_fwd)'''
            #LOGGING STUFF END

            if(len(old_error_lst_fwd) > 10): #manage list of old error vals for smoothing
                #print(old_error_lst_fwd)
                old_error_lst_fwd.pop(0)
            old_error_lst_fwd.append(int(fwd_error))

            if(len(old_error_lst_side) > 10):
                old_error_lst_side.pop(0)
            old_error_lst_side.append(int(side_error))

            #debug print
            print("fwd_err: %i, side_err: %i, output_fwd: %i, output_side: %i, fwd_d: %i, side_d: %i, fwd_i: %i, side_i %i, old_error_fwd: %i, old_error_side: %i, old_time_rn: %i"%(fwd_error, side_error, output_fwd, output_side, fwd_d, side_d, fwd_i, side_i, old_error_fwd, old_error_side, old_time_rn))
            
        time.sleep(0.05)


def sequencer_step(step_dir, *args): #changes the pid targets  dymanically. # arg of 0 means second param is abs step, arg of +/- 1 increments
    global sequence, current_step, z_trg, x_trg, y_trg, xy_reset, launch_land
    if(step_dir != 0):
        current_step += step_dir
    else:
        current_step = args[0]
    if(current_step + 1 > len(sequence)):
        #current_step = len(sequence) - 1
        launch_land = -1
    if(current_step < 0):
        current_step = 0
    if(current_step < len(sequence) ):
        x_trg = sequence[current_step][0]
        y_trg = sequence[current_step][1]
        z_trg = sequence[current_step][2]
        xy_reset = True
    print("sequencer callled! new step: %i"%(current_step))

if __name__ == "__main__":

    format = "%(asctime)s: %(message)s"

    logging.basicConfig(format=format, level=logging.INFO,

                        datefmt="%H:%M:%S")

    start_time = time.time()
    logging.info("Starting arduino...")

    #Serial takes these two parameters: serial device and baudrate
    arduino = serial.Serial(loc, 115200)


    time.sleep(2)
    logging.info("Starting threads...")



    read_arduino_thread = threading.Thread(target=read_lines_in, args=(1,))
    get_controller_thread = threading.Thread(target=read_controller, args=(2,))
    update_drone_thread = threading.Thread(target=update_drone, args=(3,))
    track_drone_thread = threading.Thread(target=track_drone, args=(4,))
    pid_vertical_thread = threading.Thread(target=do_pid_vertical, args=(5,))
    graph_thread = threading.Thread(target=graph, args=(6,))
    pid_heading_thread = threading.Thread(target=do_pid_heading, args=(7,))
    pid_xy_thread = threading.Thread(target=do_pid_xy, args=(7,))
    
    read_arduino_thread.start()
    get_controller_thread.start()
    update_drone_thread.start()
    track_drone_thread.start()
    pid_vertical_thread.start()
    graph_thread.start()
    pid_heading_thread.start()
    pid_xy_thread.start()

    logging.info("Started threads")

    # x.join()

    logging.info("Main    : all done")
