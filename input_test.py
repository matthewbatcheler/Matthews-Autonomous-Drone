
from __future__ import print_function


from inputs import get_gamepad



def main():
    x = 0
    y = 0
    z = 0
    t = 0
    while True:
        events = get_gamepad()
        for event in events:
            if(event.ev_type != "Sync" and event.ev_type != "Absolute"):
                print(event.ev_type, event.code, event.state)
               #print(x,y,z,t)
               #print(event.code)
            if(event.code == "ABS_Y"):
                #print("asb y")
                y = int(event.state)
                #print(y)
            if(event.code == "ABS_X"):
                x = int(event.state)
            if(event.code == "ABS_RZ"):
                z = int(event.state)
            if(event.code == "ABS_THROTTLE"):
                t = int(event.state)
            
                
        


if __name__ == "__main__":
    main()