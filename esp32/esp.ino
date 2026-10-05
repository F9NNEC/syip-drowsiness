#include <HardwareSerial.h>

#define BUZZER_PIN 23
#include <TinyGPSPlus.h>


TinyGPSPlus gps;
#define gpsSerial Serial2

// Variable untuk timer non-blocking
unsigned long lastDisplayTime = 0;
const unsigned long DISPLAY_INTERVAL = 2000;

void displayLocationInfo();

void setup() {
    Serial.begin(115200); 
    pinMode(BUZZER_PIN, OUTPUT);

    gpsSerial.begin(9600, SERIAL_8N1, 16, 17);
    Serial.println(F("System Ready. Waiting for GPS fix and satellites..."));
}

void loop() {
    if (Serial.available()) {
        int signal = Serial.read();
        
        if (signal == 1 || signal == 2) {
            digitalWrite(BUZZER_PIN, HIGH);
        } else {
            digitalWrite(BUZZER_PIN, LOW);
        }
    }

  // Jangan masukkan delay() di dalam atau sekitar loop ini!
  while (gpsSerial.available() > 0) {
    gps.encode(gpsSerial.read());
  }

  if (millis() - lastDisplayTime >= DISPLAY_INTERVAL) {
    lastDisplayTime = millis();

    if (gps.location.isValid()) {
      displayLocationInfo();
    } else if (millis() > 5000 && gps.charsProcessed() < 10) {
      Serial.println(F("Warning: No GPS detected. Check wiring (TX -> GPIO 16, RX -> GPIO 17) or power source."));
    } else {
      Serial.print(F("Searching for satellites... Total bytes processed: "));
      Serial.println(gps.charsProcessed());
    }
  }
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