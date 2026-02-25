import time
import numpy as np
import pyspacemouse
import threading
from typing import List
from autolab_core import RigidTransform
from autolab_core.transformations import euler_matrix
from UR5E.ur5e import UR5ERobot

class SpaceMouseRobotController:
    def __init__(self):
        self.current_action = np.zeros(6)
        self.button_0_pressed = False
        self.button_1_pressed = False
        self.gripper_state = 1
        # launch a new listener thread to listen to SpaceMouse
        self.thread = threading.Thread(target=self.read_device)
        self.thread.daemon = True
        self.thread.start()


    def read_device(self):

        button_arr = [pyspacemouse.ButtonCallback(0, self.button_0), # this is called whenever the button state changes and if button 0 is being pressed after the state change
                    pyspacemouse.ButtonCallback(1, self.button_1), # this is called whenever the button state changes and if button 1 is being pressed after the state change
                    pyspacemouse.ButtonCallback([0, 1], self.button_0_1), ] # this is called whenever the button state changes and if both buttons is being pressed after the state change

        # device = pyspacemouse.open(dof_callback=pyspacemouse.print_state, button_callback=self.someButton, button_callback_arr=button_arr)
        device = pyspacemouse.open(button_callback=self.someButton, button_callback_arr=button_arr)
        if device is not None:
            while True:
                self.state = device.read() # state: {t,x,y,z,pitch,yaw,roll,button} namedtuple
                # print(state.x, state.y, state.z, state.roll, state.pitch, state.yaw)
                self.current_action = np.array([self.state.x, self.state.y, self.state.z, self.state.roll, self.state.pitch, self.state.yaw])
        else:
            print("Failed to open spacemouse device")

    def someButton(self, state, buttons):
        """
        This function is called whenever the button state changes. This is called before the button callback array
        """
        print("Some button")

    def button_0(self, state, buttons, pressed_buttons):
        """
        state: SpaceNavigator(t=93844.164859684, x=0, y=0, z=0, roll=0, pitch=0, yaw=0, buttons=[1, 0])
        buttons: [1, 0] # whether each button is pressed
        pressed_buttons= 0 # index of buttons that is pressed
        """
        print("Opening gripper")
        self.button_0_pressed = True
        if self.gripper_state == 0:
            self.gripper_state = 1
        else:
            self.gripper_state = 0
        time.sleep(0.1999)
        self.button_0_pressed = False

    def button_1(self, state, buttons, pressed_buttons):
        """
        state: SpaceNavigator(t=93844.164859684, x=0, y=0, z=0, roll=0, pitch=0, yaw=0, buttons=[1, 0])
        buttons: [1, 0] # whether each button is pressed
        pressed_buttons= 1 # index of buttons that is pressed
        """
        print("Button 1 pressed")
        self.button_1_pressed = True
        time.sleep(0.2999)
        self.button_1_pressed = False


    def button_0_1(self, state, buttons, pressed_buttons):
        print("Buttons 0 and 1 are pressed")



def servo_pose(self,target,time=0.002,lookahead_time=0.1,gain=300):
        '''
        target in Rigidtransform being converted to: x,y,z,rx,ry,rz
        '''
        pos = RT2UR(target)
        print(f"Moving to >> Translation: {pos[:3]} | Rotation: {pos[3:]}")
        self.ur_c.servoL(pos, 0.0, 0.0, time, lookahead_time, gain)

def RT2UR(rt:RigidTransform):
    '''
    converts from rigidtransform pose to the UR format pose (x,y,z,rx,ry,rz)
    '''
    pos = rt.translation.tolist() + rt.axis_angle.tolist()
    return pos


def get_pose(self):
    p = self.ur_r.getActualTCPPose()
    return UR2RT(p)

def UR2RT(pose:List):
    '''
    converts UR format pose to RigidTransform
    '''
    m = RigidTransform.rotation_from_axis_angle(pose[-3:])
    return RigidTransform(translation=pose[:3],rotation=m)