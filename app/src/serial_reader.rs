use std::io::{self, prelude::*};
use serialport::*;

pub fn read_serial() -> io::Result<()> {
    
    // Configure serial port
    let ports = available_ports().expect("No ports found!");
    println!("Availible Ports:");
    for p in ports {
        println!("{}", p.port_name);
    }

    // Write some data to the serial port.
    /*
    let data = b"Hello, world!\n";
    port.write_all(data)?;

    // Read data from the serial port.
    let mut serial_buf: Vec<u8> = vec![0; 100];
    let num_bytes_read = port.read(&mut serial_buf)?;
    let received_data = &serial_buf[..num_bytes_read];
    println!("Received data: {:?}", received_data);
    */

    Ok(())
}

