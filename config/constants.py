"""
Configuration file for POZYX SETUP
"""
from pypozyx import (
    Coordinates, 
    DeviceCoordinates
)

REMOTE_IDS = [0x683F]

ANCHOR_CONFIG = {
    "ILS_9H": [
        DeviceCoordinates(0x685C, 1, Coordinates(10306, 8351, 2400)),
        DeviceCoordinates(0x6837, 1, Coordinates(9220, 0, 2400)),  # Move up higher
        DeviceCoordinates(0x6840, 1, Coordinates(17, 3, 2400)),
        DeviceCoordinates(0x6863, 1, Coordinates(2517, 9931, 2400)),  # Move up higher
        DeviceCoordinates(0x684B, 1, Coordinates(800, 4500, 2400)),
        DeviceCoordinates(0x1139, 1, Coordinates(9255, 2813, 2400)),
        DeviceCoordinates(0x1101, 1, Coordinates(5080, 3370, 2400)),
        DeviceCoordinates(0x1149, 1, Coordinates(6530, 7400, 2400)),
        DeviceCoordinates(0x117B, 1, Coordinates(5650, 0, 2350))
    ],
    "ILS_8H": [
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
    ],
    "GR_8H": [
        DeviceCoordinates(0x1149, 1, Coordinates(10659, 12784, 2539)),
        DeviceCoordinates(0x684B, 1, Coordinates(3989, 6654, 2515)),
        DeviceCoordinates(0x685C, 1, Coordinates(107, 0, 2550)),
        DeviceCoordinates(0x6863, 1, Coordinates(4372, 12783, 2500)),
        DeviceCoordinates(0x6840, 1, Coordinates(4800, 12861, 2478)),
        DeviceCoordinates(0x1139, 1, Coordinates(14, 7215, 2220)),
        DeviceCoordinates(0x1101, 1, Coordinates(8362, 6418, 2500)),
        DeviceCoordinates(0x6837, 1, Coordinates(4527, 9765, 2500))
    ]
}
