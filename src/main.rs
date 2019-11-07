use std::time::{Duration, Instant};

use ocl::{builders::BufferBuilder, enums::MemInfo, Buffer, ProQue};

#[macro_use]
extern crate clap;
use clap::{App, Arg};

#[macro_use]
extern crate prettytable;
use prettytable::Table;

use uom::si::f32::*;
use uom::si::information::{byte, megabyte};
use uom::si::time::{millisecond, second};

extern crate pbr;
use pbr::ProgressBar;

fn timed(end_power: usize, tries: usize) -> ocl::Result<()> {
    let kernel_src = "";
    let mut table = Table::new();
    table.add_row(row![
        "Iteration",
        "Datasize (bytes)",
        "Time (ms)",
        "Allocations/s",
        "Allocation bandwidth (MB/s)"
    ]);
    let mut queue = ProQue::builder();
    queue.src(kernel_src);
    let mut pb = ProgressBar::new((end_power + 1) as u64);
    pb.format("╢▌▌░╟");

    for i in 0..=end_power {
        let data_size: usize = 1 << i;
        let ocl_pq = queue.dims(data_size).build()?;

        let mut allocation_duration = Duration::new(0, 0);
        let mut upload_duration = Duration::new(0, 0);
        let mut download_duration = Duration::new(0, 0);
        let mut host_memory = vec![0u8; data_size];
        for _j in 0..tries {
            {
                let mut _buffer: Buffer<u8> = ocl_pq.create_buffer()?;
                let start = Instant::now();
                //_buffer.write(&host_memory).enq()?;
                _buffer.read(&mut host_memory).enq()?;
                allocation_duration += start.elapsed();
            }
        }
        let seconds = allocation_duration.as_secs_f32();
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
        pb.inc();
    }
    pb.finish_print("Finished test");
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
    println!("Running {} tests...", (end_power + 1));

    match timed(end_power, tries) {
        Ok(_) => (),
        Err(err) => println!("{}", err),
    }
}
