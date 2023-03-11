use std::io::{self, prelude::*};
use serialport::*;
use std::time::Duration;
use std::ffi::CStr;

pub fn read_serial() -> io::Result<()> {
    
    // Configure serial port
    let ports = available_ports().expect("No ports found!");
    let num_ports = ports.len();

    println!("Availible Ports: {}", num_ports);
    for p in &ports {
        println!("{}", p.port_name);
    }
    
    if num_ports > 0 {

        let mut port = serialport::new(&ports[0].port_name, 115_200)
        .timeout(Duration::from_millis(5000))
        .open().expect("Failed to open port");

        // Read data from the serial port.
        // let mut serial_buf: Vec<u8> = vec![0; 40];

        // Write some data to the serial port.
        let data = b"ping";
        port.write_all(data)?;

        /*
        let num_bytes_read = port.read(&mut serial_buf)?;

        // convert data to a string 
        let data_string = String::from_utf8(serial_buf)
            .expect("Failed to convert bytes to string")
            .split('\0')
            .next()
            .expect("No null character found when parsing")
            .to_owned();
        
        println!("Received {} bytes, data: {:?}", num_bytes_read, data_string);
        */

        let mut buffer = Vec::new();
        loop {
            let mut serial_buf = [0; 64];
            let bytes_read = port.read(&mut serial_buf)?;
            buffer.extend_from_slice(&serial_buf[..bytes_read]);
            
            // println!("Reading: {:?}", &serial_buf);
            if buffer.contains(&b'\n') {
                let data_string = String::from_utf8_lossy(&buffer);
                let null_index = data_string.find('\n').unwrap();

                let data = &data_string[..null_index];
                println!("Received data: {:?}", data);

                buffer = buffer[null_index + 1..].to_vec();
                if buffer.is_empty() {
                    break;
                }
            }
        }

    }

    Ok(())
}

