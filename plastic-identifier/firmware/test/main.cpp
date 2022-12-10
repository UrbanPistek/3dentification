/*
    Test Arduino Firmware
    Run with: 
        pio run --target upload && pio device monitor
*/

#include <Arduino.h>
#include <SPI.h>
#include <Wire.h>

// Custom drivers
#include "ads1256.h"
#include "tlc59208.h"

// Protobuf definitions
// #include <pb_encode.h>
// #include <pb_decode.h>
// #include <firmware.pb.h>

// Scan scan_msg = Scan_init_zero;
// scan_msg.led1 = 0.55;

String str;
float test_scan_data[8] = {0.04962, 0.0, 0.30735, 0.30883, 0.16934, 0.24132, 0.0, 0.0};
float test_read_adc = 0.61932;
String test_response = "Ping";

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(1);

  // Serial.print("Nanopb Msg: ")
  // Serial.println(scan_msg.led1);

  Serial.println("Initialized...");
}

void loop() {
  while (!Serial.available());
  
  str = Serial.readString();
  
  if (str == "scan"){
  
    for (int i = 0; i < 8; i++){
      Serial.print(test_scan_data[i]);
    }
    
  } else if (str == "adc"){
    Serial.print(test_read_adc);
  } else if (str == "ping") {
    Serial.print(test_response);
  }

  // Wait for outgoing data to be sent
  Serial.flush();
  
}