use serialport::*;
use std::time::Duration;
use std::error::Error;

use serde_json;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct SensorData {
    led0: f32,
    led0_var: f32,
    led0_ambient: f32,
    led1: f32,
    led1_var: f32,
    led1_ambient: f32,
    led2: f32,
    led2_var: f32,
    led2_ambient: f32,
    led3: f32,
    led3_var: f32,
    led3_ambient: f32,
    led4: f32,
    led4_var: f32,
    led4_ambient: f32,
    led5: f32,
    led5_var: f32,
    led5_ambient: f32,
    led6: f32,
    led6_var: f32,
    led6_ambient: f32,
    led7: f32,
    led7_var: f32,
    led7_ambient: f32,
}

pub fn get_scan_data() -> std::result::Result<SensorData, Box<dyn Error>> {
    
    // Configure serial port
    let ports = available_ports().expect("No ports found!");
    let num_ports = ports.len();

    if num_ports > 0 {

        let mut port = serialport::new(&ports[0].port_name, 115_200)
        .timeout(Duration::from_millis(5000))
        .open().expect("Failed to open port");

        // Write some data to the serial port.
        let data = b"ping";
        port.write_all(data).unwrap();
        
        // Wake up device
        let mut buffer = Vec::new();
        loop {
            let mut serial_buf = [0; 64];
            let bytes_read = port.read(&mut serial_buf).unwrap();
            buffer.extend_from_slice(&serial_buf[..bytes_read]);
            
            if buffer.contains(&b'\n') {
                let data_string = String::from_utf8_lossy(&buffer);
                let null_index = data_string.find('\n').unwrap();
                let data = &data_string[..null_index];
                
                if data.contains("PlasticIdentifier is initialized!") {
                    break;
                } else {
                    return Err("board failed to initialize".into());
                }

            }
        }

        // Get the spectra scan data 
        let data = b"gen_spectra";
        port.write_all(data).unwrap();

        // Store scan data in json format
        let json_data: SensorData;

        buffer = Vec::new();
        loop {
            let mut serial_buf = [0; 1024];
            let bytes_read = port.read(&mut serial_buf).unwrap();
            buffer.extend_from_slice(&serial_buf[..bytes_read]);
            
            if buffer.contains(&b'}') {
                let data_string = String::from_utf8_lossy(&buffer);
                let null_index = data_string.find('}').unwrap() + 1;

                let data = &data_string[..null_index];
                json_data = serde_json::from_str(&data).unwrap();
                break;

           }
        }

        // return json data
        return Ok(json_data);
    } else {

        return Err("failed to get scan".into());
    }
}

