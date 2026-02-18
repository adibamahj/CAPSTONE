import sys
import time

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

BEAM_ARM_DELAY = 2  # seconds


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def toggle_estop():
    global estop
    
    estop = not estop
    
    if estop:
        print("\n=== E-STOP ACTIVATED ===")
        print("Motion and inventory disabled.")
    else:
        print("\n=== E-STOP CLEARED ===")
        print("System ready.")


def quit_program():
    print("\nExiting program...")
    sys.exit(0)


def home():
    global homed, current_bin, last_selected_bin
    
    if estop:
        print("E-STOP active. Clear it before homing.")
        return
    
    print("\n=== HOMING ROUTINE ===")
    time.sleep(1)
    homed = True
    current_bin = 0
    last_selected_bin = 0
    print("Homing complete. Now at BIN0.")


def move_to_bin(target):
    global current_bin, last_selected_bin
    
    if estop:
        print("E-STOP active. Motion disabled.")
        return
    
    if not homed:
        print("ERROR: Not homed.")
        return
    
    if inventory_pending:
        print("INVENTORY REQUIRED: Press 'i' before selecting another bin.")
        return
    
    if gate_lockout:
        print("GATE LOCKOUT: Bin removed or waiting 2s after reinsertion.")
        return
    
    steps = (target - current_bin) % 4
    
    if steps == 0:
        print(f"Already at BIN{target}")
        return
    
    print(f"Moving CCW {steps} bin(s) to BIN{target}...")
    time.sleep(1)
    
    current_bin = target
    last_selected_bin = target
    print(f"Now at BIN{current_bin}")
    
    simulate_gate_block()


def simulate_gate_block():
    global gate_blocked, gate_lockout
    
    gate_blocked = True
    gate_lockout = True
    print("GATE: BLOCKED (bin pulled out).")
    print("Type 'push' to simulate reinserting bin.")


def simulate_gate_reinsert():
    global gate_blocked, gate_lockout, inventory_pending
    
    if estop:
        print("E-STOP active. Operation blocked.")
        return
    
    if not gate_blocked:
        print("Gate already open.")
        return
    
    print("GATE: OPEN. Waiting 2 seconds...")
    time.sleep(BEAM_ARM_DELAY)
    
    gate_blocked = False
    gate_lockout = False
    inventory_pending = True
    
    print("GATE READY. Press 'i' to perform inventory sensing.")


def run_inventory():
    global inventory_pending
    
    if estop:
        print("E-STOP active. Inventory disabled.")
        return
    
    if not inventory_pending:
        print("Nothing to inventory.")
        return
    
    print("\n=== INVENTORY SENSING ===")
    print("Rotating 2 bins (180°)...")
    time.sleep(1)
    
    print("Measuring distance (simulated)...")
    time.sleep(2)
    
    simulated_distance = 10.5
    
    if simulated_distance < 12.0:
        print("HI")
    else:
        print("LO")
    
    print(f"(Distance = {simulated_distance} cm)")
    
    inventory_pending = False
    print("Inventory complete. You may select another bin.")


def print_status():
    print("\n--- STATUS ---")
    print(f"Homed: {homed}")
    print(f"Current Bin: {current_bin}")
    print(f"Gate Blocked: {gate_blocked}")
    print(f"Inventory Pending: {inventory_pending}")
    print(f"E-STOP Active: {estop}")
    print("----------------\n")


# ============================================================
# STARTUP
# ============================================================

print("Power-up: Homing required.")
print("Type 'h' to home or 'q' to quit.")

while not homed:
    cmd = input("> ").strip().lower()
    
    if cmd == "h":
        home()
    elif cmd == "e":
        toggle_estop()
    elif cmd == "q":
        quit_program()
    else:
        print("Type 'h' to home, 'e' to toggle E-STOP, or 'q' to quit.")

print("\nCommands: h, bin0-3, i, push, status, e (toggle), q\n")


# ============================================================
# MAIN LOOP
# ============================================================

while True:
    cmd = input("> ").strip().lower()
    
    if cmd == "q":
        quit_program()
    
    elif cmd == "e":
        toggle_estop()
    
    elif cmd == "h":
        home()
    
    elif cmd in ["bin0", "bin1", "bin2", "bin3"]:
        target = int(cmd[-1])
        move_to_bin(target)
    
    elif cmd == "push":
        simulate_gate_reinsert()
    
    elif cmd == "i":
        run_inventory()
    
    elif cmd == "status":
        print_status()
    
    else:
        print("Unknown command. Use h, bin0-3, i, push, status, e, q.")
