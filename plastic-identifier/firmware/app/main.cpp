/*
    Main Arduino Firmware
    Run with: 
        pio run --target upload && pio device monitor
*/
#include <Arduino.h>
#include <SPI.h>
#include <Wire.h>

#include "assert.h"
#include "cli.h"
#include "ads1256.h"
#include "tlc59208.h"

// JSON Serialization
#include "arduinojson.h"

#define CLI_MODE false
#define SERIAL_MODE !CLI_MODE
#define DEVELOP_MODE false

static const int CLKSPEED_MHZ = 7.68;
static const float VREF = 2.5;

ADS1256 adc(CLKSPEED_MHZ, VREF, false);
TLC59208 ledctrl;

// Scan data doc
StaticJsonDocument<128> scan_doc;
StaticJsonDocument<128> calib_doc;
StaticJsonDocument<128> spectrum_doc;

// Read from serial
String serial_str;

/*
    Functions that run through the CLI interface. 
*/
#if CLI_MODE
    Cli cli;

    void cli_scan(int argc, char *argv[])
    {
        Serial.println("Performing Scan...");
        float readings[8] = {0};
        for (int i=0; i<8; i++) {
            ledctrl.on(i);
            delay(5);
            adc.waitDRDY(); 
            readings[i] = adc.readCurrentChannel();
            ledctrl.off(i);
        }

        for (int i=0; i<8; i++) {
            Serial.print("Led: ");
            Serial.print(i);
            Serial.print(", ");
            Serial.print(readings[i], 5);
            Serial.println(" mV");
        }
        Serial.println();
    }

    void cli_read_adc(int argc, char *argv[])
    {
        Serial.println("Reading ADC...");
        Serial.print("pinDRDY: ");
        Serial.println(digitalRead(pinDRDY));
        
        adc.waitDRDY(); 
        Serial.println("adc.waitDRDY();");
        
        float val = adc.readCurrentChannel();
        Serial.println("==> Channel Read");
        Serial.println(val , 5);
    }

    void cli_led(int argc, char *argv[])
    {
        int num;        // Parameter 1: led number [0..7]
        bool state;     // Parameter 2: led state [on/off]
        if (argc != 3) {
            Serial.println("Usage: led <number> <on/off>");
            return;
        }

        // Parameter checking
        bool args_ok = true;
        num = (int)strtol(argv[1], NULL, 10);
        if (num < 0 || num > 7) args_ok = false;
        if      (strcmp(argv[2], "on") == 0) state = true;
        else if (strcmp(argv[2], "off") == 0) state = false;
        else args_ok = false;

        if (args_ok == false) {
            Serial.println("Usage: Usage: led <number> <on/off>");
        } else {
            state == true ? ledctrl.on(num) : ledctrl.off(num);
        }
    }

    void cli_help(int argc, char *argv[])
    {
        cli.list_commands();
    }

#endif
/*
    Functions that run through the Serial interface. 
*/
#if SERIAL_MODE

    void ping(void)
    {
        Serial.print("pong");
    }

    void scan()
    {
        float readings[8] = {0};
        for (int i=0; i<8; i++) {
            ledctrl.on(i);
            delay(5);
            adc.waitDRDY(); 
            readings[i] = adc.readCurrentChannel();
            ledctrl.off(i);
        }

        for (int i=0; i<8; i++) {
            // set key value
            String key = "led" + String(i+1);
            scan_doc[key] = readings[i];
        }
        serializeJson(scan_doc, Serial);
    }

    void calibrate()
    {
        // LED scatter
        float scatter_readings[8] = {0};
        for (int i=0; i<8; i++) {
            ledctrl.on(i);
            delay(5);
            adc.waitDRDY();
            scatter_readings[i] = adc.readCurrentChannel();
            
            // set key value
            String key = "led" + String(i+1);
            calib_doc[key] = scatter_readings[i];
        }
    }

    void gen_spectra(uint8_t exposure_time_ms, uint8_t num_readings, bool ambient)
    {
        // TODO: check if LED scatter calibration was done
        // _______________________________________________

        // initial ambient light
        for (int i=0; i<8; i++) {
            ledctrl.off(i);
        }
        delay(exposure_time_ms);
        adc.waitDRDY();
        calib_doc["ambient0"] = adc.readCurrentChannel(); 

        // scan
        float ambient_readings[8] = {0};
        float scan_readings[8] = {0};
        for (int n=0; n<num_readings; n++) {
            for (int i=0; i<8; i++) {
                // read reflectance from LED
                ledctrl.on(i);
                delay(exposure_time_ms);
                adc.waitDRDY(); 
                scan_readings[i] = adc.readCurrentChannel();

                // log reflectance reading
                String key1 = "rdg" + String(n+1) + "led" + String(i+1);
                scan_doc[key1] = scan_readings[i];

                if (ambient) {
                    // read ambient light
                    ledctrl.off(i);
                    delay(exposure_time_ms);
                    adc.waitDRDY();
                    ambient_readings[i] = adc.readCurrentChannel();

                    // log ambient reading
                    String key2 = "rdg" + String(n+1) + "ambient" + String(i+1);
                    calib_doc[key2] = ambient_readings[i];
                }
            }
        }

        serializeJson(scan_doc, Serial);
        serializeJson(calib_doc, Serial);
        return;
    }

    void read(int ledNumber)
    {
        float reading = 0;

        ledctrl.on(ledNumber);
        delay(5000);
        adc.waitDRDY(); 
        reading = adc.readCurrentChannel();
        ledctrl.off(ledNumber);

        delay(25);
        Serial.print(reading);
    }

#endif

void setup()
{
    #if CLI_MODE
        Serial.begin(9600);

    #endif
    #if SERIAL_MODE
        Serial.begin(115200);
    #endif

    #if !DEVELOP_MODE

        SPI.begin();
        Wire.begin();
        ledctrl.begin();
        adc.begin(ADS1256_DRATE_30000SPS, ADS1256_GAIN_1, false); 
        adc.setChannel(0,1);    // differential ADC reading 

    #endif

    #if CLI_MODE
        
        cli.add_command({"scan", cli_scan, "Perform a scan sequence: for each led measure adc value"});
        cli.add_command({"adc", cli_read_adc, "Reads ADC measurement"});
        cli.add_command({"led", cli_led, "Turns an LED <number> on/off <state>.\n\t\t\t\tUsage: led <number> <state>"});
        cli.add_command({"help", cli_help, "Lists all available commands"});
        cli.begin();

    #endif
    #if SERIAL_MODE

        // Initialize calibration doc
        // led scatter
        calib_doc["led1"] = 0.0;
        calib_doc["led2"] = 0.0;
        calib_doc["led3"] = 0.0;
        calib_doc["led4"] = 0.0;
        calib_doc["led5"] = 0.0;
        calib_doc["led6"] = 0.0;
        calib_doc["led7"] = 0.0;
        calib_doc["led8"] = 0.0;
        
        // ambient light
        calib_doc["ambient0"] = 0.0; // initial ambient light
        calib_doc["ambient1"] = 0.0; // ambient light after led1 scan
        calib_doc["ambient2"] = 0.0; // ambient light after led2 scan
        calib_doc["ambient3"] = 0.0; // ...
        calib_doc["ambient4"] = 0.0;
        calib_doc["ambient5"] = 0.0;
        calib_doc["ambient6"] = 0.0;
        calib_doc["ambient7"] = 0.0;
        calib_doc["ambient8"] = 0.0;

        // Initialize scan doc
        scan_doc["led1"] = 0.0;
        scan_doc["led2"] = 0.0;
        scan_doc["led3"] = 0.0;
        scan_doc["led4"] = 0.0;
        scan_doc["led5"] = 0.0;
        scan_doc["led6"] = 0.0;
        scan_doc["led7"] = 0.0;
        scan_doc["led8"] = 0.0;

    #endif

    Serial.println("PlasticIdentifier is initialized!");
}

#if CLI_MODE
    
    void loop()
    {
        cli.handle();
    }

#endif
#if SERIAL_MODE

    void loop()
    {
        while (!Serial.available());

        serial_str = Serial.readString();
        
        if (serial_str == "ping")
        {
            ping();
        }
        else if (serial_str == "scan")
        {
            // run scan
            scan();
        }
        else if (serial_str == "gen_spectrum")
        {
            // generate spectrum
            gen_spectrum();
        }
        else if (serial_str == "read")
        {
            // wait for user to input which LED to read
            while (Serial.available() == 0);

            int ledNumber = Serial.parseInt();

            // Serial.print(ledNumber);
            read(ledNumber);
        }

        // Wait for outgoing data to be sent
        Serial.flush();
    }

#endif
