// serial module
mod dnir_reader;

use dnir_reader::*;

fn main() {
    println!("\nDNIR Application...");
    
    let scan_json = get_scan_data().unwrap();
    println!("Scan: \n{:#?}", scan_json);
}
