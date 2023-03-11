// serial module
mod serial_reader;

use serial_reader::*;

fn main() {
    println!("DNIR Application...");
    
    read_serial();
}
