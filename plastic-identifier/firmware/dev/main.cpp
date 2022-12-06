/*
    Testing Firmware
    Run with: 
        pio run --target upload && pio device monitor
*/
#include <ads1256.h>
#include <SPI.h>
#include <Wire.h>

float clockMHZ = 7.68; // crystal frequency used on ADS1256
float vRef = 2.5; // voltage reference

// Construct and init ADS1256 object
ADS1256 adc(clockMHZ, vRef, false); // RESETPIN is permanently tied to 3.3v

float sensor1, sps;
long lastTime, currentTime, elapsedTime; 
int counter; 

void setup()
{
    Serial.begin(9600);
    SPI.begin();
    Wire.begin();
  
    Serial.println("Starting ADC");
    
    adc.begin(ADS1256_DRATE_100SPS, ADS1256_GAIN_1, false); 

    Serial.println("ADC Started");
    
    // Set MUX Register to AINO-AIN1 so it start doing the ADC conversion
    adc.setChannel(0,1);
}

void loop()
{ 
    Serial.print("Start pinDRDY: ");
    Serial.println(digitalRead(pinDRDY));

    currentTime = millis();
    elapsedTime = currentTime - lastTime; 
    if (elapsedTime >= 1000){ 
        sps = counter*1.0/elapsedTime*1000;
        lastTime = currentTime; 
        counter=0;    
    }  
    adc.waitDRDY(); // wait for DRDY to go low before changing multiplexer register 
    sensor1 = adc.readCurrentChannel(); // DOUT arriving here is from MUX AIN0 and AIN8 (GND)

    counter++; 
    Serial.print("Counter: ");
    Serial.print(counter);
    Serial.print(" SR: ");
    Serial.print(sps);  
    Serial.print(" Value : ");
    Serial.println(sensor1, 5);   // print with 2 decimals

    // delay 100ms
    delay(100);
    Serial.print("After Read pinDRDY: ");
    Serial.println(digitalRead(pinDRDY));
    
    // adc.setChannel(0,1);
    // adc.sendCommand(ADS1256_CMD_STANDBY);
    // delay(500);
    // Serial.print("After Standby pinDRDY: ");
    // Serial.println(digitalRead(pinDRDY));

    // adc.sendCommand(ADS1256_CMD_WAKEUP);
    // delay(500);
    // Serial.print("After Wakeup pinDRDY: ");
    // Serial.println(digitalRead(pinDRDY));

    // adc.sendCommand(ADS1256_CMD_SYNC);
    // delay(500);
    // Serial.print("After Sync pinDRDY: ");
    // Serial.println(digitalRead(pinDRDY));
}
