#include <HardwareSerial.h>
#include <TinyGPSPlus.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <WiFi.h>

#define BUZZER_PIN 23

#define BUTTON_UP 25
#define BUTTON_ENTER 26
#define BUTTON_DOWN 27

TinyGPSPlus gps;
#define gpsSerial Serial2

LiquidCrystal_I2C lcd(0x27, 16, 2);

const char* ssid = "adit";
const char* password = "11111111";

unsigned long lastDisplayTime = 0;
const unsigned long DISPLAY_INTERVAL = 2000;

bool paused = false;
int cursor = 0;
int gpsPage = 0;
int screenPage = 0;

void setup() {
  Serial.begin(115200);

  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(BUTTON_UP, INPUT_PULLUP);
  pinMode(BUTTON_ENTER, INPUT_PULLUP);
  pinMode(BUTTON_DOWN, INPUT_PULLUP);

  gpsSerial.begin(9600, SERIAL_8N1, 16, 17);

  WiFi.begin(ssid, password);

  lcd.init();
  lcd.backlight();

  updateLCD();

  Serial.println("System Ready. Waiting for GPS fix and satellites...");
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

  while (gpsSerial.available() > 0) {
    gps.encode(gpsSerial.read());
  }

  if (digitalRead(BUTTON_UP) == LOW) {
    if (screenPage == 1) {
      screenPage = 0;
      cursor = 0;
    } else {
      cursor = 0;
    }

    updateLCD();
    delay(200);
  }

  if (digitalRead(BUTTON_DOWN) == LOW) {
    if (screenPage == 0 && cursor == 0) {
      cursor = 1;
    } else if (screenPage == 0 && cursor == 1) {
      screenPage = 1;
      cursor = 0;
    } else {
      screenPage = 0;
      cursor = 0;
    }

    updateLCD();
    delay(200);
  }

  if (digitalRead(BUTTON_ENTER) == LOW) {
    if (screenPage == 0 && cursor == 0) {
      paused = !paused;

      if (paused) {
        beepPause();
      } else {
        startupSound();
      }

      updateLCD();
    } else if (screenPage == 0 && cursor == 1) {
      calibrateSystem();
    }

    delay(200);
  }

  if (millis() - lastDisplayTime >= DISPLAY_INTERVAL) {
    lastDisplayTime = millis();

    if (gps.location.isValid()) {
      gpsPage++;

      if (gpsPage > 2) {
        gpsPage = 0;
      }
    }

    updateLCD();

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

void updateLCD() {
  if (screenPage == 0) {
    lcd.setCursor(0, 0);
    lcd.print(cursor == 0 ? "> " : "  ");
    lcd.print(paused ? "START" : "PAUSE");
    lcd.print("         ");

    lcd.setCursor(0, 1);
    lcd.print(cursor == 1 ? "> " : "  ");
    lcd.print("KALIBRASI");
    lcd.print("       ");
  } else {
    lcd.setCursor(0, 0);

    if (gps.location.isValid()) {
      if (gpsPage == 0) {
        lcd.print("LAT:");
        lcd.print(gps.location.lat(), 4);
      } else if (gpsPage == 1) {
        lcd.print("LNG:");
        lcd.print(gps.location.lng(), 4);
      } else {
        lcd.print("SPD:");
        lcd.print(gps.speed.kmph(), 1);
        lcd.print(" km/h");
      }
    } else {
      lcd.print("GPS Searching");
    }

    lcd.print("   ");

    lcd.setCursor(0, 1);
    lcd.print("WIFI ");
    lcd.print(WiFi.status() == WL_CONNECTED
              ? "Connected"
              : "Disconnected");
    lcd.print("  ");
  }
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
  lcd.clear();
  lcd.setCursor(2, 0);
  lcd.print("Kalibrasi...");
  
  Serial.println("Kalibrasi dimulai.");

  delay(1000);

  lcd.clear();
  lcd.setCursor(2, 0);
  lcd.print("Kalibrasi");
  lcd.setCursor(2, 1);
  lcd.print("Selesai");

  Serial.println("Kalibrasi selesai.");

  delay(1000);

  updateLCD();
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