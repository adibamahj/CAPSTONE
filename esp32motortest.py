import serial, time, re

class JetsonESP32Controller:
    
    STATES = ["BOOT", "HOMING", "IDLE", "MOVING", 
              "GATE_LOCKOUT", "INVENTORY_PENDING", "ESTOP"]
    
    def __init__(self, port="/dev/ttyTHS1", baud=115200):
        self.ser = serial.Serial(port, baud, timeout=1)
        self.state = "BOOT"
        self.current_bin = -1
        self.last_distance_cm = None

    # ── Send a newline-terminated command ──────────────────────────
    def send_cmd(self, cmd: str):
        self.ser.write((cmd.strip() + "\n").encode())

    # ── Non-blocking line reader ───────────────────────────────────
    def read_line(self, timeout=10.0) -> str | None:
        deadline = time.time() + timeout
        buf = ""
        while time.time() < deadline:
            if self.ser.in_waiting:
                c = self.ser.read().decode(errors="ignore")
                if c == "\n":
                    return buf.strip()
                buf += c
        return None

    # ── Monitor ESP32 output and update Jetson state ───────────────
    def monitor_response(self, timeout=15.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.read_line(timeout=1.0)
            if line is None:
                continue

            print(f"[ESP32] {line}")

            # State transitions based on ESP32 output
            if "Homing complete" in line:
                self.state = "IDLE"
                self.current_bin = 0
                return "HOMED"

            elif "Done. Now at BIN" in line:
                self.current_bin = int(line[-1])
                self.state = "GATE_LOCKOUT"  # anticipate user pulling bin
                return "BIN_SELECTED"

            elif "GATE: BLOCKED" in line:
                self.state = "GATE_LOCKOUT"

            elif "GATE: Ready. Press 'i'" in line:
                self.state = "INVENTORY_PENDING"
                return "INVENTORY_PENDING"

            elif "Distance(cm) =" in line:
                match = re.search(r"Distance\(cm\)\s*=\s*([\d.]+)", line)
                if match:
                    self.last_distance_cm = float(match.group(1))
                    result = "HI" if self.last_distance_cm > 12.0 else "LO"
                    print(f"[JETSON] Inventory result: {result} ({self.last_distance_cm:.1f} cm)")
                    self.state = "IDLE"
                    return result

            elif "E-STOP" in line:
                self.state = "ESTOP"
                return "ESTOP"

            elif "INVENTORY REQUIRED" in line or "inventoryPending" in line:
                self.state = "INVENTORY_PENDING"
                return "INVENTORY_PENDING"

        return "TIMEOUT"

    # ── High-level operations ──────────────────────────────────────
    def home(self):
        assert self.state in ["BOOT", "IDLE"], "Can only home from BOOT or IDLE"
        self.send_cmd("h")
        return self.monitor_response(timeout=60)

    def select_bin(self, bin_num: int):
        assert self.state == "IDLE", f"Cannot select bin in state {self.state}"
        assert 0 <= bin_num <= 3
        self.send_cmd(f"bin{bin_num}")
        return self.monitor_response(timeout=30)

    def run_inventory(self):
        assert self.state == "INVENTORY_PENDING", "Inventory not pending"
        self.send_cmd("i")
        return self.monitor_response(timeout=20)  # 4s sensor + buffer

    def emergency_stop(self):
        self.send_cmd("e")
        self.state = "ESTOP"

    # ── Full dispense workflow (called by LLM/UI layer) ───────────
    def dispense(self, bin_num: int) -> dict:
        """
        Full workflow: select bin → wait for user to pull/replace → 
        inventory check → return HI/LO stock level
        """
        if self.state == "BOOT":
            result = self.home()
            if result != "HOMED":
                return {"success": False, "error": "Homing failed"}

        result = self.select_bin(bin_num)
        if result != "BIN_SELECTED":
            return {"success": False, "error": f"Move failed: {result}"}

        # Wait for gate lockout to clear (user interaction - long timeout)
        print("[JETSON] Waiting for user to pull and re-insert bin...")
        gate_result = self.monitor_response(timeout=120)  
        if gate_result != "INVENTORY_PENDING":
            return {"success": False, "error": f"Gate timeout: {gate_result}"}

        inv_result = self.run_inventory()  # "HI" or "LO"
        
        return {
            "success": True,
            "bin": bin_num,
            "stock_level": inv_result,          # "HI" or "LO"
            "distance_cm": self.last_distance_cm
        }