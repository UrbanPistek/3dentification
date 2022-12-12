int x;
String str;

float test_scan_data[8] = {0.04962, 0.0, 0.30735, 0.30883, 0.16934, 0.24132, 0.0, 0.0};
float test_read_adc = 0.61932;
String test_response = "Ping";

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(1);
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
