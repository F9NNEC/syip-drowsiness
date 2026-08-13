import serial
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        self.connect()

    def connect(self):
        """Connect to ESP32."""
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=1)
            logger.info(f"✓ ESP32 terhubung di port {self.port}")
        except Exception as e:
            logger.warning(f"✗ ESP32 tidak ditemukan di {self.port}. Error: {e}")
            self.serial = None

    def is_connected(self):
        """Check if ESP32 is connected."""
        return self.serial is not None and self.serial.is_open

    def send(self, signal):
        """
        Send signal to ESP32.
        
        Args:
            signal: 1 (drowsy/alarm) atau 0 (alert/normal)
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

    def send_alert(self):
        """Send alert signal (0)."""
        return self.send(0)

    def disconnect(self):
        """Disconnect from ESP32."""
        if self.serial is not None:
            try:
                self.serial.close()
                logger.info("ESP32 disconnected")
            except:
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

