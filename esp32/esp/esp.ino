#include <HardwareSerial.h>
#include <TinyGPSPlus.h>
#include <WiFi.h>
#include <esp_now.h>

struct SendData {
  int32_t heartRate;  // BPM (0 = belum ada hasil)
  int32_t spo2;       // SpO2 % (0 = belum ada hasil)
  int32_t pause;      // 1 = Pause aktif, 0 = tidak (level, tetap 1 selama tombol toggle aktif)
  int32_t calibrate;  // 1 selama ~1 detik setelah tombol Calibrate ditekan, lalu 0
};

#define BUZZER_PIN 23

#define PRINT_MAC_AT_BOOT 0

TinyGPSPlus gps;
#define gpsSerial Serial2

unsigned long lastBootMessage = 0;
const unsigned long BOOT_RETRY_INTERVAL = 1000;
const unsigned long HOST_TIMEOUT = 5000;
const int HOST_HEARTBEAT_SIGNAL = 6;

unsigned long lastDisplayTime = 0;
const unsigned long DISPLAY_INTERVAL = 1000;  // sesuaikan dengan kodemu

bool paused = false;
bool calibrationRunning = false;
bool systemStarted = false;
unsigned long lastHostHeartbeat = 0;
int alarmSignal = 0;
int cursor = 0;
int gpsPage = 0;
int screenPage = 0;

portMUX_TYPE espMux = portMUX_INITIALIZER_UNLOCKED;
volatile bool vitalsBaru = false;     
volatile int32_t watchBpm = 0;
volatile int32_t watchSpo2 = 0;
volatile int32_t watchPause = 0;       
volatile bool calibrateRequest = false; 
int32_t lastCalibrateRx = 0;         

void onDataReceive(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
  if (len != sizeof(SendData)) return;

  SendData rx;
  memcpy(&rx, data, sizeof(rx));

  portENTER_CRITICAL(&espMux);
  watchBpm = rx.heartRate;
  watchSpo2 = rx.spo2;
  watchPause = rx.pause;
  if (rx.calibrate == 1 && lastCalibrateRx == 0) calibrateRequest = true;
  lastCalibrateRx = rx.calibrate;
  vitalsBaru = true;
  portEXIT_CRITICAL(&espMux);
}

void setup() {
  Serial.begin(115200);

  pinMode(BUZZER_PIN, OUTPUT);

  gpsSerial.begin(9600, SERIAL_8N1, 16, 17);

  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
#if PRINT_MAC_AT_BOOT
  Serial.print("MAC:");
  Serial.println(WiFi.macAddress());
#endif
  if (esp_now_init() == ESP_OK) {
    esp_now_register_recv_cb(onDataReceive);
  } else {
    Serial.println("ESPNOW: init gagal");
  }

  Serial.println("CMD:BOOT");
  lastBootMessage = millis();
}

void loop() {
  while (Serial.available()) {
    int signal = Serial.read();

    if (signal == 5) {
      Serial.println("CMD:RESTARTING");
      delay(100);
      ESP.restart();
    } else if (signal == 4) {
      systemStarted = true;
      paused = false;
      calibrationRunning = false;
      alarmSignal = 0;
      screenPage = 0;
      cursor = 0;
      lastHostHeartbeat = millis();
      Serial.println("CMD:READY");
    } else if (signal == HOST_HEARTBEAT_SIGNAL && systemStarted) {
      lastHostHeartbeat = millis();
    } else if (signal == 3 && calibrationRunning) {
      calibrationRunning = false;
    } else if (systemStarted && (signal == 1 || signal == 2)) {
      if (signal != alarmSignal) {
        tone(BUZZER_PIN, signal == 1 ? 2000 : 2500);
        alarmSignal = signal;
      }
    } else if (signal == 0) {
      noTone(BUZZER_PIN);
      digitalWrite(BUZZER_PIN, LOW);
      alarmSignal = 0;
    }
  }

  if (systemStarted && millis() - lastHostHeartbeat >= HOST_TIMEOUT) {
    systemStarted = false;
    paused = true;
    calibrationRunning = false;
    screenPage = 0;
    cursor = 0;
    noTone(BUZZER_PIN);
    digitalWrite(BUZZER_PIN, LOW);
    alarmSignal = 0;
    Serial.println("CMD:HOST_TIMEOUT");
  }

  if (!systemStarted && millis() - lastBootMessage >= BOOT_RETRY_INTERVAL) {
    lastBootMessage = millis();
    Serial.println("CMD:BOOT");
  }

  while (gpsSerial.available() > 0) {
    gps.encode(gpsSerial.read());
  }

  // Data dari smartwatch: Pause, Calibrate, BPM, SpO2
  espNowLayani();

  if (millis() - lastDisplayTime >= DISPLAY_INTERVAL) {
    lastDisplayTime = millis();

    if (gps.location.isValid()) {
      gpsPage++;

      if (gpsPage > 2) {
        gpsPage = 0;
      }
    }

    if (gps.location.isValid()) {
      displayLocationInfo();
    } else if (millis() > 5000 && gps.charsProcessed() < 10) {
      Serial.println(
        "Warning: No GPS detected. Check wiring "
        "(TX -> GPIO 16, RX -> GPIO 17) or power source."
      );
    } else {
      Serial.print("Searching for satellites... Total bytes processed: ");
      Serial.println(gps.charsProcessed());
    }
  }
}

void espNowLayani() {
  int32_t bpm, spo2, pauseReq;
  bool baru, kalibrasi;

  portENTER_CRITICAL(&espMux);
  baru = vitalsBaru;
  vitalsBaru = false;
  kalibrasi = calibrateRequest;
  calibrateRequest = false;
  bpm = watchBpm;
  spo2 = watchSpo2;
  pauseReq = watchPause;
  portEXIT_CRITICAL(&espMux);

  if (baru) kirimVitals(bpm, spo2);

  if (systemStarted) {
    bool mintaPause = (pauseReq == 1);
    if (mintaPause != paused) {
      paused = mintaPause;

      if (paused) {
        beepPause();
      } else {
        startupSound();
      }

      Serial.println(paused ? "CMD:PAUSE" : "CMD:START");
    }

    if (kalibrasi) {
      calibrateSystem();
    }
  }
}

void kirimVitals(int32_t bpm, int32_t spo2) {
  Serial.printf("{\"bpm\": %ld, \"spo2\": %ld}\n", (long)bpm, (long)spo2);
}

void beepPause() {
  tone(BUZZER_PIN, 2000);
  delay(150);
  noTone(BUZZER_PIN);
}

void startupSound() {
  tone(BUZZER_PIN, 1000);
  delay(150);

  tone(BUZZER_PIN, 1500);
  delay(150);

  tone(BUZZER_PIN, 2000);
  delay(500);

  noTone(BUZZER_PIN);
}

void calibrateSystem() {
  calibrationRunning = true;
  Serial.println("CMD:CALIBRATE");
}

void displayLocationInfo() {
  if (gps.location.isValid()) {
    Serial.print("{\"lat\": ");
    Serial.print(gps.location.lat(), 6);

    Serial.print(", \"lng\": ");
    Serial.print(gps.location.lng(), 6);

    Serial.print(", \"speed\": ");
    Serial.print(gps.speed.kmph());

    Serial.println("}");
  }
}