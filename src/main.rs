use std::time::{Duration, Instant};

use ocl::{Buffer, ProQue, enums::MemInfo};

#[macro_use]
extern crate clap;
use clap::{App, Arg};

#[macro_use]
extern crate prettytable;
use prettytable::Table;

use uom::si::f32::*;
use uom::si::information::{byte, megabyte};
use uom::si::time::{millisecond, second};

fn timed(end_power: usize, tries: usize) -> ocl::Result<()> {
    let kernel_src = "";
    let mut table = Table::new();
    table.add_row(row![
        "Iteration",
        "Datasize (bytes)",
        "Time (ms)",
        "Operations/s",
        "Allocation bandwidth"
    ]);
    let mut queue = ProQue::builder();
    queue.src(kernel_src);

    for i in 0..=end_power {
        println!("{}", i);
        let data_size: usize = 1 << i;

        let ocl_pq = queue.dims(data_size).build()?;

        let mut duration = Duration::new(0, 0);
        for _j in 0..tries {
            {
                let kern_start = Instant::now();
                let _buffer: Buffer<u8> = ocl_pq.create_buffer()?;
                //println!("{:?}", _buffer.mem_info(MemInfo::Type));
                duration += kern_start.elapsed();
            }
        }
        let seconds = duration.as_secs_f32();
        let seconds_per_try = seconds / tries as f32;
        let time = Time::new::<second>(seconds / tries as f32);
        let bytes = Information::new::<byte>(data_size as f32);
        let bandwidth = bytes.get::<megabyte>() / time.get::<second>();
        table.add_row(row![
            i,
            data_size,
            time.get::<millisecond>(),
            1.0 / seconds_per_try,
            bandwidth
        ]);
    }
    table.printstd();

    Ok(())
}

pub fn main() {
    let matches = App::new("GPU bandwidth test")
        .version(crate_version!())
        .author(crate_authors!())
        .about("Measure GPU bandwidth")
        .arg(
            Arg::with_name("tries")
                .short("n")
                .help("The number of tries for each test")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("end-power")
                .short("e")
                .help("At which power of two to stop the test")
                .takes_value(true),
        )
        .get_matches();

    let tries = value_t!(matches, "tries", usize).unwrap_or(50);
    let end_power = value_t!(matches, "end-power", usize).unwrap_or(30);
    println!("Number of tries per test: {}", tries);

    match timed(end_power, tries) {
        Ok(_) => (),
        Err(err) => println!("{}", err),
    }
}
