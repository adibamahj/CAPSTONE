import sys
import time
import serial
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

SERIAL_PORT = "/dev/ttyUSB0"       
BAUD_RATE = 115200
LOG_FILE = "inventory_log.txt"

BEAM_ARM_DELAY = 2  # seconds


# ============================================================
# STATE VARIABLES
# ============================================================

homed = False
current_bin = None
last_selected_bin = None

gate_blocked = False
gate_lockout = False
inventory_pending = False

estop = False


# ============================================================
# SERIAL SETUP
# ============================================================

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # allow ESP32 reset
    print("Connected to ESP32.")
except:
    print("ERROR: Could not open serial port.")
    sys.exit(1)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def send_to_esp(cmd):
    ser.write((cmd + "\n").encode())


def log_inventory(result):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"{timestamp} | BIN{last_selected_bin} | {result}\n"
    
    with open(LOG_FILE, "a") as f:
        f.write(entry)
    
    print(f"Logged: {entry.strip()}")


def emergency_stop():
    global estop
    estop = True
    send_to_esp("e")
    print("\nE-STOP ACTIVATED.")
    sys.exit(0)


def home():
    global homed, current_bin, last_selected_bin
    
    send_to_esp("h")
    print("Homing...")
    
    wait_for_serial_message("Homing complete")
    
    homed = True
    current_bin = 0
    last_selected_bin = 0
    print("Now at BIN0.")


def move_to_bin(target):
    global current_bin, last_selected_bin
    
    if not homed:
        print("ERROR: Not homed.")
        return
    
    if inventory_pending:
        print("INVENTORY REQUIRED: Press 'i' before selecting another bin.")
        return
    
    if gate_lockout:
        print("GATE LOCKOUT active.")
        return
    
    cmd = f"bin{target}"
    send_to_esp(cmd)
    
    current_bin = target
    last_selected_bin = target
    print(f"Selecting BIN{target}...")


def run_inventory():
    global inventory_pending
    
    if not inventory_pending:
        print("Nothing to inventory.")
        return
    
    send_to_esp("i")
    print("Waiting for HI/LO from ESP32...")
    
    result = wait_for_hi_lo()
    
    if result:
        log_inventory(result)
        inventory_pending = False
    else:
        print("No valid inventory result received.")


def wait_for_hi_lo():
    """
    Waits until ESP32 sends HI or LO.
    """
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode().strip()
            
            if line == "HI" or line == "LO":
                print(f"ESP32: {line}")
                return line
            
            # Optional: detect gate ready message
            if "Ready" in line:
                global inventory_pending
                inventory_pending = True
                print("Inventory now required.")


def wait_for_serial_message(keyword):
    """
    Waits until a serial line contains a keyword.
    """
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode().strip()
            print("ESP32:", line)
            if keyword in line:
                return


# ============================================================
# STARTUP
# ============================================================

print("Power-up: Homing required.")
print("Type 'h' to home.")

while not homed:
    cmd = input("> ").strip().lower()
    
    if cmd == "h":
        home()
    elif cmd == "e":
        emergency_stop()
    else:
        print("Please type 'h' to home.")


print("\nCommands: h, bin0-3, i, e\n")


# ============================================================
# MAIN LOOP
# ============================================================

while True:
    cmd = input("> ").strip().lower()
    
    if cmd == "e":
        emergency_stop()
    
    elif cmd == "h":
        home()
    
    elif cmd in ["bin0", "bin1", "bin2", "bin3"]:
        target = int(cmd[-1])
        move_to_bin(target)
    
    elif cmd == "i":
        run_inventory()
    
    else:
        print("Unknown command.")
