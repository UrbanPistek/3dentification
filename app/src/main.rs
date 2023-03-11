// serial module
mod serial_reader;

use serial_reader::*;

fn main() {
    println!("\nDNIR Application...");
    
    read_serial();
}
