import virtual_keyboard as k
import VirtualMouse as m
import time


def call_mouse():    
    print("switching to mouse")
    time.sleep(1)
    m.mouse()

def call_keyboard():
    print("switching to keyboard")
    time.sleep(1)
    k.keyboard()

