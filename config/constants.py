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
        DeviceCoordinates(0x685C, 1, Coordinates(10306, 8351, 2400)), #Front Door
        DeviceCoordinates(0x6840, 1, Coordinates(9220, 0, 2400)), #Toilet
        DeviceCoordinates(0x6837, 1, Coordinates(17, 3, 2400)), #Back of bedroom
        DeviceCoordinates(0x6863, 1, Coordinates(2517, 9931, 2400)), #Desk
        DeviceCoordinates(0x684B, 1, Coordinates(800, 4500, 2400)), #TV
        DeviceCoordinates(0x1139, 1, Coordinates(9255, 2813, 2400)), #Kitchen
        DeviceCoordinates(0x117B, 1, Coordinates(4400, 0, 2400)), #Front of Bedroom              
        DeviceCoordinates(0x1149, 1, Coordinates(6530, 7400, 2400)), #Living Room Table  
        DeviceCoordinates(0x1101, 1, Coordinates(5080, 3370, 2400)), #Couch
    ],
    "ILS_8H": [
        DeviceCoordinates(0x685C, 1, Coordinates(10306, 8351, 2400)),
        DeviceCoordinates(0x6840, 1, Coordinates(9220, 0, 2400)),  
        DeviceCoordinates(0x6837, 1, Coordinates(17, 3, 2400)),
        DeviceCoordinates(0x6863, 1, Coordinates(2517, 9931, 2400)),  
        DeviceCoordinates(0x684B, 1, Coordinates(800, 4500, 2400)),
        DeviceCoordinates(0x1139, 1, Coordinates(9255, 2813, 2400)),
        DeviceCoordinates(0x117B, 1, Coordinates(5650, 0, 2350)),
        DeviceCoordinates(0x1149, 1, Coordinates(6530, 7400, 2400)),
        # DeviceCoordinates(0x1101, 1, Coordinates(5080, 3370, 2400)) 
    ],
     "ILS_7H": [
        DeviceCoordinates(0x685C, 1, Coordinates(10306, 8351, 2400)),
        DeviceCoordinates(0x6840, 1, Coordinates(9220, 0, 2400)),  
        DeviceCoordinates(0x6837, 1, Coordinates(17, 3, 2400)),
        DeviceCoordinates(0x6863, 1, Coordinates(2517, 9931, 2400)),  
        DeviceCoordinates(0x684B, 1, Coordinates(800, 4500, 2400)),
        DeviceCoordinates(0x1139, 1, Coordinates(9255, 2813, 2400)),
        DeviceCoordinates(0x117B, 1, Coordinates(5650, 0, 2350)),
        # DeviceCoordinates(0x1149, 1, Coordinates(6530, 7400, 2400)),
        # DeviceCoordinates(0x1101, 1, Coordinates(5080, 3370, 2400)) 
    ],
     "ILS_6H": [
        DeviceCoordinates(0x685C, 1, Coordinates(10306, 8351, 2400)),
        DeviceCoordinates(0x6840, 1, Coordinates(9220, 0, 2400)),  
        DeviceCoordinates(0x6837, 1, Coordinates(17, 3, 2400)),
        DeviceCoordinates(0x6863, 1, Coordinates(2517, 9931, 2400)),  
        DeviceCoordinates(0x684B, 1, Coordinates(800, 4500, 2400)),
        DeviceCoordinates(0x1139, 1, Coordinates(9255, 2813, 2400)),
        # DeviceCoordinates(0x117B, 1, Coordinates(5650, 0, 2350)),
        # DeviceCoordinates(0x1149, 1, Coordinates(6530, 7400, 2400)),
        # DeviceCoordinates(0x1101, 1, Coordinates(5080, 3370, 2400)) 
    ],
     "ILS_5H": [
        DeviceCoordinates(0x685C, 1, Coordinates(10306, 8351, 2400)),
        DeviceCoordinates(0x6840, 1, Coordinates(9220, 0, 2400)),  
        DeviceCoordinates(0x6837, 1, Coordinates(17, 3, 2400)),
        DeviceCoordinates(0x6863, 1, Coordinates(2517, 9931, 2400)),  
        DeviceCoordinates(0x684B, 1, Coordinates(800, 4500, 2400)),
        # DeviceCoordinates(0x1139, 1, Coordinates(9255, 2813, 2400)),
        # DeviceCoordinates(0x117B, 1, Coordinates(5650, 0, 2350)),
        # DeviceCoordinates(0x1149, 1, Coordinates(6530, 7400, 2400)),
        # DeviceCoordinates(0x1101, 1, Coordinates(5080, 3370, 2400)) 
    ],
     "ILS_4H": [
        DeviceCoordinates(0x685C, 1, Coordinates(10306, 8351, 2400)),
        DeviceCoordinates(0x6840, 1, Coordinates(9220, 0, 2400)),  
        DeviceCoordinates(0x6837, 1, Coordinates(17, 3, 2400)),
        DeviceCoordinates(0x6863, 1, Coordinates(2517, 9931, 2400)),  
        # DeviceCoordinates(0x684B, 1, Coordinates(800, 4500, 2400)),
        # DeviceCoordinates(0x1139, 1, Coordinates(9255, 2813, 2400)),
        # DeviceCoordinates(0x117B, 1, Coordinates(5650, 0, 2350)),
        # DeviceCoordinates(0x1149, 1, Coordinates(6530, 7400, 2400)),
        # DeviceCoordinates(0x1101, 1, Coordinates(5080, 3370, 2400)) 
    ],

}
