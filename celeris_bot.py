import http.client
import json
import logging
import math
import os
import queue
import threading
import time
import urllib.request
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
alert_queue = queue.Queue()
worker_lock = threading.Lock()
worker_started = False
last_alert_queued_at = None
ALERT_COOLDOWN_SECONDS = 5.0

ALERT_MESSAGES = {
    "drowsy": "PERINGATAN: Pengemudi terdeteksi mengantuk.",
    "danger": "BAHAYA: Pengemudi terdeteksi dalam kondisi berbahaya.",
}


def _telegram_request(bot_token, method, payload):
    url = f"https://api.telegram.org/bot{bot_token}/{method}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=10) as response:
        result = json.loads(response.read().decode("utf-8"))

    if not result.get("ok"):
        raise RuntimeError(result.get("description", "Telegram API request failed"))


def _valid_coordinates(gps_data):
    latitude = gps_data.get("lat")
    longitude = gps_data.get("lng")
    if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
        return None
    if not math.isfinite(latitude) or not math.isfinite(longitude):
        return None
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        return None
    return latitude, longitude


def _send_alert(bot_token, chat_id, status, gps_data):
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    coordinates = _valid_coordinates(gps_data)
    if coordinates is None:
        text = f"{ALERT_MESSAGES[status]}\nWaktu: {timestamp}\nLokasi GPS belum tersedia."
    else:
        latitude, longitude = coordinates
        text = (
            f"{ALERT_MESSAGES[status]}\n"
            f"Waktu: {timestamp}\n"
            f"Koordinat: {latitude:.6f}, {longitude:.6f}\n"
            f"Peta: https://maps.google.com/?q={latitude:.6f},{longitude:.6f}"
        )

    try:
        _telegram_request(bot_token, "sendMessage", {"chat_id": chat_id, "text": text})
    except (http.client.HTTPException, OSError, ValueError, RuntimeError) as error:
        logger.error("Gagal mengirim notifikasi Telegram: %s", error)
        return

    if coordinates is not None:
        try:
            latitude, longitude = coordinates
            _telegram_request(
                bot_token,
                "sendLocation",
                {"chat_id": chat_id, "latitude": latitude, "longitude": longitude},
            )
        except (http.client.HTTPException, OSError, ValueError, RuntimeError) as error:
            logger.error("Gagal mengirim lokasi GPS Telegram: %s", error)


def _send_alerts():
    while True:
        bot_token, chat_id, status, gps_data = alert_queue.get()
        try:
            _send_alert(bot_token, chat_id, status, gps_data)
        finally:
            alert_queue.task_done()


def send_detection_alert(status, gps_data):
    """Queue a Telegram alert for a drowsy or dangerous detection."""
    global worker_started, last_alert_queued_at

    if status not in ALERT_MESSAGES:
        raise ValueError(f"Unsupported Telegram alert status: {status}")

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        logger.error("Set TELEGRAM_BOT_TOKEN dan TELEGRAM_CHAT_ID di file .env.")
        return False

    with worker_lock:
        current_time = time.monotonic()
        if (last_alert_queued_at is not None
                and current_time - last_alert_queued_at < ALERT_COOLDOWN_SECONDS):
            logger.info("Notifikasi Telegram dilewati: jeda minimum 5 detik belum terpenuhi.")
            return False

        if not worker_started:
            threading.Thread(target=_send_alerts, daemon=True).start()
            worker_started = True

        alert_queue.put((bot_token, chat_id, status, dict(gps_data)))
        last_alert_queued_at = current_time

    return True
