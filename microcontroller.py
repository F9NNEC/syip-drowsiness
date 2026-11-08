import serial
import logging
import threading
import json
import queue
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERIAL_PARTIAL_LINE_TIMEOUT_SECONDS = 5
MAX_SERIAL_LINE_BYTES = 1024
HOST_HEARTBEAT_INTERVAL_SECONDS = 1
HOST_HANDSHAKE_RETRY_SECONDS = 1
SERIAL_RECONNECT_DELAY_SECONDS = 2

gps_lock = threading.Lock()
last_gps = {"lat": None, "lng": None, "speed": None}
vitals_lock = threading.Lock()
last_vitals = {"bpm": None, "spo2": None}


def get_gps_data():
    with gps_lock:
        return dict(last_gps)


def get_vitals_data():
    with vitals_lock:
        return dict(last_vitals)


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
        self.write_lock = threading.Lock()
        self.handshake_pending = False
        self.commands = queue.Queue()
        self.connect()

    def connect(self):
        """Start the serial reader, which retries connection until stopped."""
        self.running = True
        self.reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.reader_thread.start()
        print(f"\n{'='*50}")
        print("TERMINAL SERIAL MONITOR")
        print(f"{'='*50}\n")

    def _read_loop(self):
        """Read incoming data from ESP32 and print it in terminal."""
        pending = bytearray()
        pending_since = None
        discard_until_newline = False

        while self.running:
            if not self.is_connected():
                try:
                    connection = serial.Serial(self.port, self.baudrate, timeout=0.2)
                    connection.reset_input_buffer()
                    self.serial = connection
                    logger.info(f"✓ ESP32 terhubung di port {self.port}")
                    pending.clear()
                    pending_since = None
                    discard_until_newline = False
                    self.handshake_pending = True
                    next_handshake = time.monotonic()
                    next_heartbeat = next_handshake
                except Exception as e:
                    self.serial = None
                    logger.warning(
                        "ESP32 belum terhubung di %s; mencoba lagi dalam %s detik. Error: %s",
                        self.port,
                        SERIAL_RECONNECT_DELAY_SECONDS,
                        e
                    )
                    time.sleep(SERIAL_RECONNECT_DELAY_SECONDS)
                    continue

            try:
                now = time.monotonic()
                if self.handshake_pending and now >= next_handshake:
                    if not self.send(4):
                        logger.error("Gagal mengirim handshake mulai ke ESP32")
                        raise serial.SerialException("Handshake write failed")
                    next_handshake = now + HOST_HANDSHAKE_RETRY_SECONDS
                elif not self.handshake_pending and now >= next_heartbeat:
                    if not self.send(6):
                        logger.error("Gagal mengirim heartbeat ke ESP32")
                        raise serial.SerialException("Heartbeat write failed")
                    next_heartbeat = now + HOST_HEARTBEAT_INTERVAL_SECONDS

                chunk = self.serial.readline()
                now = time.monotonic()

                if not chunk:
                    if (pending and pending_since is not None
                            and now - pending_since >= SERIAL_PARTIAL_LINE_TIMEOUT_SECONDS):
                        logger.warning(
                            "Membuang data serial ESP32 yang tidak lengkap "
                            "setelah %s detik",
                            SERIAL_PARTIAL_LINE_TIMEOUT_SECONDS
                        )
                        pending.clear()
                        pending_since = None
                        discard_until_newline = True
                    continue

                remaining = chunk
                while remaining:
                    if discard_until_newline:
                        newline = remaining.find(b'\n')
                        if newline < 0:
                            break
                        remaining = remaining[newline + 1:]
                        discard_until_newline = False
                        continue

                    newline = remaining.find(b'\n')
                    if newline < 0:
                        if not pending:
                            pending_since = now
                        pending.extend(remaining)
                        if len(pending) > MAX_SERIAL_LINE_BYTES:
                            logger.warning(
                                "Membuang baris serial ESP32 yang melebihi %s byte",
                                MAX_SERIAL_LINE_BYTES
                            )
                            pending.clear()
                            pending_since = None
                            discard_until_newline = True
                        break

                    line = bytes(pending) + remaining[:newline]
                    pending.clear()
                    pending_since = None
                    remaining = remaining[newline + 1:]
                    self._process_line(line)
            except Exception as e:
                logger.error(f"Error reading ESP32 data: {e}")
                connection = self.serial
                self.serial = None
                if connection is not None:
                    try:
                        connection.close()
                    except serial.SerialException as close_error:
                        logger.error("Error closing ESP32 serial port: %s", close_error)
                pending.clear()
                pending_since = None
                discard_until_newline = False
                time.sleep(SERIAL_RECONNECT_DELAY_SECONDS)

    def _process_line(self, line):
        text = line.decode('utf-8', errors='replace').strip()
        if not text:
            return
        if '\ufffd' in text:
            logger.warning("Membuang baris serial ESP32 dengan karakter tidak valid")
            return

        timestamp = datetime.now().strftime("%H:%M:%S")

        if text.startswith('{'):
            try:
                data = json.loads(text)
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                logger.warning("Baris JSON serial ESP32 tidak valid: %s", e)
                return

            try:
                lat = data.get('lat')
                lng = data.get('lng')
                speed = data.get('speed')
                if lat is not None and lng is not None:
                    with gps_lock:
                        last_gps['lat'] = float(lat)
                        last_gps['lng'] = float(lng)
                        last_gps['speed'] = float(speed) if speed is not None else None
            except (AttributeError, TypeError, ValueError) as e:
                logger.warning("Data GPS serial ESP32 tidak valid: %s", e)

            for key in ('bpm', 'spo2'):
                if key not in data or data[key] is None:
                    continue
                try:
                    value = int(data[key])
                except (TypeError, ValueError, OverflowError) as e:
                    logger.warning("Data %s serial ESP32 tidak valid: %s", key, e)
                    continue
                with vitals_lock:
                    last_vitals[key] = value
        elif text.startswith('$'):
            print(f"[{timestamp}] GPS: {text}")
        elif text == 'CMD:BOOT':
            logger.info("CMD:BOOT diterima; handshake startup akan dikirim")
            self.handshake_pending = True
        elif text == 'CMD:READY':
            self.handshake_pending = False
            logger.info("Handshake ESP32 selesai")
            print(f"[{timestamp}] ESP32: sistem dashboard siap")
        elif text in ('CMD:START', 'CMD:PAUSE', 'CMD:CALIBRATE'):
            self.commands.put(text[4:])
        else:
            print(f"[{timestamp}] ESP32: {text}")

    def is_connected(self):
        """Check if ESP32 is connected."""
        return self.serial is not None and self.serial.is_open

    def send(self, signal):
        """
        Send signal to ESP32.
        
        Args:
            signal: 0 (normal), 1 (drowsy), 2 (danger), atau 3 (kalibrasi selesai)
        """
        if not self.is_connected():
            return False
        
        try:
            with self.write_lock:
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

    def restart(self):
        """Request a software restart of the ESP32."""
        if not self.send(5):
            logger.error("Gagal mengirim perintah restart ke ESP32")
            return False
        logger.info("Perintah restart dikirim ke ESP32")
        return True

    def read_commands(self):
        """Return control commands received from ESP32."""
        commands = []
        while True:
            try:
                commands.append(self.commands.get_nowait())
            except queue.Empty:
                return commands

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


def init_esp32(port='COM3'):
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
