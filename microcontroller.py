import serial
import logging
import threading
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

gps_lock = threading.Lock()
last_gps = {"lat": None, "lng": None, "speed": None}


def get_gps_data():
    with gps_lock:
        return dict(last_gps)


class ESP32Connection:
    """serial connection handler."""
    
    def __init__(self, port='COM3', baudrate=115200):
        """
        Initialize ESP32 connection.
        port: Serial port (COM, /dev/ttyUSB0)
        """
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.reader_thread = None
        self.running = False
        self.connect()

    def connect(self):
        """Connect to ESP32."""
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=1)
            self.running = True
            self.reader_thread = threading.Thread(target=self._read_loop, daemon=True)
            self.reader_thread.start()
            logger.info(f"✓ ESP32 terhubung di port {self.port}")
            print(f"\n{'='*50}")
            print(f"  📡 TERMINAL SERIAL MONITOR")
            print(f"{'='*50}\n")
        except Exception as e:
            logger.warning(f"✗ ESP32 tidak ditemukan di {self.port}. Error: {e}")
            self.serial = None
            self.running = False

    def _read_loop(self):
        """Read incoming data from ESP32 and print it in terminal."""
        while self.running and self.serial is not None:
            try:
                if self.serial.in_waiting > 0:
                    line = self.serial.readline()
                    if not line:
                        continue

                    text = line.decode('utf-8', errors='replace').strip()
                    if not text:
                        continue

                    timestamp = datetime.now().strftime("%H:%M:%S")

                    if text.startswith('{'):
                        try:
                            data = json.loads(text)
                            lat = data.get('lat')
                            lng = data.get('lng')
                            speed = data.get('speed')
                            if lat is not None and lng is not None:
                                with gps_lock:
                                    last_gps['lat'] = float(lat)
                                    last_gps['lng'] = float(lng)
                                    last_gps['speed'] = float(speed) if speed is not None else None
                        except Exception:
                            pass
                    elif text.startswith('$'):
                        print(f"[{timestamp}] GPS: {text}")
                    else:
                        print(f"[{timestamp}] ESP32: {text}")
                else:
                    # short sleep supaya thread tidak membebani CPU
                    import time
                    time.sleep(0.05)
            except Exception as e:
                logger.error(f"Error reading ESP32 data: {e}")
                break

    def is_connected(self):
        """Check if ESP32 is connected."""
        return self.serial is not None and self.serial.is_open

    def send(self, signal):
        """
        Send signal to ESP32.
        
        Args:
            signal: 0 (normal), 1 (drowsy), atau 2 (danger)
        """
        if not self.is_connected():
            return False
        
        try:
            self.serial.write(bytes([signal]))
            return True
        except Exception as e:
            logger.error(f"Error sending signal: {e}")
            return False

    def send_drowsy(self):
        """Send drowsy signal (1)."""
        return self.send(1)

    def send_danger(self):
        """Send danger signal (2)."""
        return self.send(2)

    def send_alert(self):
        """Send alert signal (0)."""
        return self.send(0)

    def disconnect(self):
        """Disconnect from ESP32."""
        self.running = False
        if self.serial is not None:
            try:
                self.serial.close()
                logger.info("ESP32 disconnected")
            except Exception:
                pass
            self.serial = None

    def __del__(self):
        self.disconnect()


# Global instance
_esp32 = None


def init_esp32(port='COM8'):
    """Initialize global ESP32 connection."""
    global _esp32
    _esp32 = ESP32Connection(port=port, baudrate=115200)
    return _esp32


def get_esp32():
    """Get ESP32 connection."""
    return _esp32


def close_esp32():
    """Close ESP32 connection."""
    global _esp32
    if _esp32:
        _esp32.disconnect()
        _esp32 = None

