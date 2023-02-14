#!/usr/bin/env python
"""
The Pozyx ready to localize tutorial (c) Pozyx Labs
Please read the tutorial that accompanies this sketch:
https://www.pozyx.io/Documentation/Tutorials/ready_to_localize/Python
This tutorial requires at least the contents of the Pozyx Ready to Localize kit. It demonstrates the positioning capabilities
of the Pozyx device both locally and remotely. Follow the steps to correctly set up your environment in the link, change the
parameters and upload this sketch. Watch the coordinates change as you move your device around!
"""
from time import sleep

from pypozyx import (POZYX_POS_ALG_UWB_ONLY, POZYX_3D, Coordinates, POZYX_SUCCESS, PozyxConstants, version,
                     DeviceCoordinates, PozyxSerial, get_first_pozyx_serial_port, SingleRegister, DeviceList,
                     PozyxRegisters, EulerAngles, Acceleration, LinearAcceleration, AngularVelocity)
from pythonosc.udp_client import SimpleUDPClient
import time

import csv
import pandas as pd
import sys
from pathlib import Path
from config import constants

info = []
if len(sys.argv) < 2:
    csv_name = input("Write a name: ")
else:
    csv_name = sys.argv[1]

data_path = Path(__file__).resolve().parents[0].joinpath("data")
data_path.mkdir(exist_ok=True)
output_file_path = data_path.joinpath(f"{csv_name}.csv")

f = open(output_file_path, 'w')


class ReadyToLocalize(object):
    """Continuously calls the Pozyx positioning function and prints its position."""

    def __init__(self, pozyx, osc_udp_client, anchors, algorithm=POZYX_POS_ALG_UWB_ONLY, dimension=POZYX_3D,
                 height=1000, remote_id=None):
        self.pozyx = pozyx
        self.osc_udp_client = osc_udp_client

        self.anchors = anchors
        self.algorithm = algorithm
        self.dimension = dimension
        self.height = height
        self.remote_id = remote_id

    def setup(self):
        """Sets up the Pozyx for positioning by calibrating its anchor list."""
        print("------------POZYX POSITIONING V{} -------------".format(version))
        print("")
        print("- System will manually configure tag")
        print("")
        print("- System will auto start positioning")
        print("")
        if self.remote_id is None:
            self.pozyx.printDeviceInfo(self.remote_id)
        else:
            for device_id in [None] + self.remote_id:
                self.pozyx.printDeviceInfo(device_id)
        print("")
        print("------------POZYX POSITIONING V{} -------------".format(version))
        print("")

        self.setAnchorsManual(save_to_flash=False)
        # self.printPublishConfigurationResult()

    def loop(self):
        """Performs positioning and displays/exports the results."""
        for tag_id in self.remote_id:
            position = Coordinates()
            self.printOrientationAcceleration(tag_id)
            status = self.pozyx.doPositioning(
                position, self.dimension, self.height, self.algorithm, remote_id=tag_id)
            if status == POZYX_SUCCESS:
                self.printPublishPosition(position, tag_id)
            else:
                self.printPublishErrorCode("positioning", tag_id)
            sleep(0.001)

    def printOrientationAcceleration(self, tag_id):
        global orientation
        global acceleration
        global linear_acceleration
        global angular_velocity
        # Changed acceleration to linear accel (Ignores gravity)
        orientation = EulerAngles()
        acceleration = Acceleration()
        linear_acceleration = LinearAcceleration()
        angular_velocity = AngularVelocity()
        self.pozyx.getEulerAngles_deg(orientation, tag_id)
        self.pozyx.getAcceleration_mg(acceleration, tag_id)
        self.pozyx.getLinearAcceleration_mg(linear_acceleration, tag_id)
        self.pozyx.getAngularVelocity_dps(angular_velocity, tag_id)
        """print("Orientation: %s, acceleration: %s" % (str(orientation), str(acceleration)))"""

    def printPublishPosition(self, position, network_id):
        """Prints the Pozyx's position and possibly sends it as a OSC packet"""
        if network_id is None:
            network_id = 0
        print(
            "POS ID {}, x(mm): {pos.x} y(mm): {pos.y} z(mm): {pos.z} Orientation: {orient} Acceleration: {accel} AngularVelocity: {ang_vel}".format(
                "0x%0.4x" % network_id, pos=position, orient=str(orientation), accel=str(linear_acceleration), ang_vel=str(angular_velocity)))
        global info
        """Separate orientation and acceleration strings into individual components"""
        orient = str(orientation)
        orient_sp = orient.split()
        heading = (orient_sp[1])[:-1]
        roll = (orient_sp[3])[:-1]
        pitch = (orient_sp[5])
        accel = str(linear_acceleration)
        accel_sp = accel.split()
        accel_x = (accel_sp[1])[:-1]
        accel_y = (accel_sp[3])[:-1]
        accel_z = accel_sp[5]
        ang_vel = str(angular_velocity)
        ang_vel_sp = ang_vel.split()
        ang_vel_x = (ang_vel_sp[1])[:-1]
        ang_vel_y = (ang_vel_sp[3])[:-1]
        ang_vel_z = (ang_vel_sp[5])

        current_time = time.time()
        info.append([current_time, position.x, position.y, position.z, heading, roll, pitch, accel_x, accel_y, accel_z,
                     ang_vel_x, ang_vel_z, ang_vel_y, "0x%0.4x" % network_id])
        df = pd.DataFrame(info)
        df.columns = ['Time', 'x', 'y', 'z', 'heading', 'roll', 'pitch', 'accel_x', 'accel_y', 'accel_z', 'angvel_x', 'angvel_y', 'angvel_z', 'tag_id']
        df.to_csv(output_file_path, mode='a', index=False, header=None)
        info = []  # empty info after saving
        if self.osc_udp_client is not None:
            self.osc_udp_client.send_message(
                "/position",
                [network_id, int(position.x), int(position.y), int(position.z), float(heading), float(roll),
                 float(pitch), float(accel_x), float(accel_y), float(accel_z), float(ang_vel_x), float(ang_vel_y), float(ang_vel_z)])

    def printPublishErrorCode(self, operation, network_id):
        """Prints the Pozyx's error and possibly sends it as a OSC packet"""
        error_code = SingleRegister()
        status = self.pozyx.getErrorCode(error_code, network_id)
        if network_id is None:
            self.pozyx.getErrorCode(error_code)
            print("LOCAL ERROR %s, %s" % (operation, self.pozyx.getErrorMessage(error_code)))
            if self.osc_udp_client is not None:
                self.osc_udp_client.send_message("/error", [operation, 0, error_code[0]])
            return
        if status == POZYX_SUCCESS:
            print("ERROR %s on ID %s, %s" %
                  (operation, "0x%0.4x" % network_id, self.pozyx.getErrorMessage(error_code)))
            if self.osc_udp_client is not None:
                self.osc_udp_client.send_message(
                    "/error", [operation, network_id, error_code[0]])
        else:
            self.pozyx.getErrorCode(error_code)
            print("ERROR %s, couldn't retrieve remote error code, LOCAL ERROR %s" %
                  (operation, self.pozyx.getErrorMessage(error_code)))
            if self.osc_udp_client is not None:
                self.osc_udp_client.send_message("/error", [operation, 0, -1])
            # should only happen when not being able to communicate with a remote Pozyx.

    def setAnchorsManual(self, save_to_flash=False):
        """Adds the manually measured anchors to the Pozyx's device list one for one."""
        for tag_id in self.remote_id:
            status = self.pozyx.clearDevices(tag_id)
            for anchor in self.anchors:
                status &= self.pozyx.addDevice(anchor, tag_id)
            if len(self.anchors) > 4:
                status &= self.pozyx.setSelectionOfAnchors(PozyxConstants.ANCHOR_SELECT_AUTO, len(self.anchors),
                                                           remote_id=tag_id)

            if save_to_flash:
                self.pozyx.saveAnchorIds(tag_id)
                self.pozyx.saveRegisters([PozyxRegisters.POSITIONING_NUMBER_OF_ANCHORS], tag_id)

    def printPublishConfigurationResult(self):
        """Prints and potentially publishes the anchor configuration result in a human-readable way."""
        list_size = SingleRegister()

        self.pozyx.getDeviceListSize(list_size, self.remote_id)
        print("List size: {0}".format(list_size[0]))
        if list_size[0] != len(self.anchors):
            self.printPublishErrorCode("configuration")
            return
        device_list = DeviceList(list_size=list_size[0])
        self.pozyx.getDeviceIds(device_list, self.remote_id)
        print("Calibration result:")
        print("Anchors found: {0}".format(list_size[0]))
        print("Anchor IDs: ", device_list)

        for i in range(list_size[0]):
            anchor_coordinates = Coordinates()
            self.pozyx.getDeviceCoordinates(device_list[i], anchor_coordinates, self.remote_id)
            print("ANCHOR, 0x%0.4x, %s" % (device_list[i], str(anchor_coordinates)))
            if self.osc_udp_client is not None:
                self.osc_udp_client.send_message(
                    "/anchor",
                    [device_list[i], int(anchor_coordinates.x), int(anchor_coordinates.y), int(anchor_coordinates.z)])
                sleep(0.025)

    def printPublishAnchorConfiguration(self):
        """Prints and potentially publishes the anchor configuration"""
        for anchor in self.anchors:
            print("ANCHOR,0x%0.4x,%s" % (anchor.network_id, str(anchor.coordinates)))
            if self.osc_udp_client is not None:
                self.osc_udp_client.send_message(
                    "/anchor", [anchor.network_id, int(anchor.coordinates.x), int(anchor.coordinates.y),
                                int(anchor.coordinates.z)])
                sleep(0.025)


if __name__ == "__main__":

    # shortcut to not have to find out the port yourself
    serial_port = get_first_pozyx_serial_port()
    if serial_port is None:
        print("No Pozyx connected. Check your USB cable or your driver!")
        quit()

    remote_id = constants.REMOTE_IDS  # remote device network ID
    remote = True  # whether to use a remote device
    if not remote:
        remote_id = None

    # enable to send position data through OSC
    use_processing = True

    # configure if you want to route OSC to outside your localhost. Networking knowledge is required.
    ip = "127.0.0.1"
    network_port = 8888

    osc_udp_client = None
    if use_processing:
        osc_udp_client = SimpleUDPClient(ip, network_port)

    # necessary data for calibration, change the IDs and coordinates yourself according to your measurement
    """8H ANCHORS"""
    anchors = [
        DeviceCoordinates(0x685C, 1, Coordinates(10306, 8351, 2400)),
        DeviceCoordinates(0x6837, 1, Coordinates(9220, 0, 2400)),  # Move up higher
        DeviceCoordinates(0x6840, 1, Coordinates(17, 3, 2400)),
        DeviceCoordinates(0x6863, 1, Coordinates(2517, 9931, 2400)),  # Move up higher
        DeviceCoordinates(0x684B, 1, Coordinates(800, 4500, 2400)),
        DeviceCoordinates(0x1139, 1, Coordinates(9255, 2813, 2400)),
        DeviceCoordinates(0x1101, 1, Coordinates(5080, 3370, 2400)),
        DeviceCoordinates(0x1149, 1, Coordinates(6530, 7400, 2400)),
        # DeviceCoordinates(0x111C, 1, Coordinates(7150, 2600, 1230)),
        # DeviceCoordinates(0x117B, 1, Coordinates(5650, 0, 2350))
    ]

    # """8L ANCHORS"""
    # anchors = [
    #     DeviceCoordinates(0x685C, 1, Coordinates(10306, 8351, 130)),
    #     DeviceCoordinates(0x6837, 1, Coordinates(9220, 0, 130)),  # Move up higher
    #     DeviceCoordinates(0x6840, 1, Coordinates(17, 3, 130)),
    #     DeviceCoordinates(0x6863, 1, Coordinates(2517, 9931, 130)),  # Move up higher
    #     DeviceCoordinates(0x684B, 1, Coordinates(800, 4500, 130)),
    #     # DeviceCoordinates(0x1139, 1, Coordinates(9604, 2082, 1000)),
    #     DeviceCoordinates(0x1101, 1, Coordinates(5080, 3370, 130)),
    #     DeviceCoordinates(0x1149, 1, Coordinates(6530, 7400, 130)),
    #     # DeviceCoordinates(0x111C, 1, Coordinates(7150, 2600, 1230)),
    #     # DeviceCoordinates(0x117B, 1, Coordinates(5650, 0, 2350))
    # ]
    # positioning algorithm to use, other is PozyxConstants.POSITIONING_ALGORITHM_TRACKING
    algorithm = PozyxConstants.POSITIONING_ALGORITHM_UWB_ONLY
    # positioning dimension. Others are PozyxConstants.DIMENSION_2D, PozyxConstants.DIMENSION_2_5D
    dimension = PozyxConstants.DIMENSION_3D
    # height of device, required in 2.5D positioning
    height = 1000

    pozyx = PozyxSerial(serial_port)
    r = ReadyToLocalize(pozyx, osc_udp_client, anchors, algorithm, dimension, height, remote_id)
    r.setup()
    while True:
        r.loop()

f.close()