"""
Run this script to find the button/axis indices of your Xbox controller.

    python tests/test_xbox_mapping.py

Press each button and move each stick/trigger to see its index and value.
Ctrl+C to exit.
"""

import os
import struct
import time

DEVICE = "/dev/input/js1"

JS_BTN  = 0x01
JS_AXIS = 0x02

fd = os.open(DEVICE, os.O_RDONLY | os.O_NONBLOCK)
print(f"Opened {DEVICE}  — press buttons / move sticks (Ctrl+C to stop)\n")

while True:
    try:
        data = os.read(fd, 8)
        ts, value, etype, number = struct.unpack("IhBB", data)
        etype &= ~0x80  # strip init flag

        if etype == JS_BTN and value == 1:
            print(f"  BUTTON PRESS   number={number}")

        elif etype == JS_AXIS and abs(value) > 3000:
            norm = value / 32767.0
            print(f"  AXIS           number={number:2d}   raw={value:7d}   norm={norm:+.3f}")

    except BlockingIOError:
        time.sleep(0.02)
    except KeyboardInterrupt:
        break

os.close(fd)
print("\nDone.")
