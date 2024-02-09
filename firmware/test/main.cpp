/*
    Test Arduino Firmware
    Run with:
        pio run --target upload
*/

#include <Arduino.h>
#include <SPI.h>
#include <Wire.h>

// Custom drivers
#include "ads1256.h"
#include "tlc59208.h"

// JSON Serialization
#include "arduinojson.h"

String str;
float test_scan_data[8] = {0.04962, 0.0, 0.30735, 0.30883, 0.16934, 0.24132, 0.0, 0.0};
float test_read_adc = 0.61932;
String test_response = "Ping";

// Scan data doc
StaticJsonDocument<128> doc;

void setup()
{
  Serial.begin(115200);
  Serial.setTimeout(1);

  // Serial.print("Nanopb Msg: ")
  // Serial.println(scan_msg.led1);

  Serial.println("Initialized...");

  // Add data to doc
  doc["led1"] = 0.005443;
  doc["led2"] = 0.352;
  doc["led3"] = 0.00324;
  doc["led4"] = 0.967;
  doc["led5"] = 0.0324;
  doc["led6"] = 0.1234;
  doc["led7"] = 0.4723;
  doc["led8"] = 0.0472369;
}

void loop()
{
  while (!Serial.available())
    ;

  str = Serial.readString();

  if (str == "scan")
  {

    // simulate a delay before data is ready
    delay(5000);

    // Generate the minified JSON and send it to the Serial port
    serializeJson(doc, Serial);
    Serial.print("\n");

  }
  else if (str == "adc")
  {
    Serial.print(test_read_adc);
  }
  else if (str == "ping")
  {
    Serial.print(test_response);
  }

  // Wait for outgoing data to be sent
  Serial.flush();
}