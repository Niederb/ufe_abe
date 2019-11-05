use std::time::{Duration, Instant};
extern crate ocl;
extern crate ocl_extras;
extern crate time;

#[macro_use]
extern crate prettytable;
use prettytable::Table;

extern crate uom;

use uom::si::f32::*;
use uom::si::information::{byte, megabyte};
use uom::si::time::{millisecond, second};

use ocl::{Buffer, ProQue};

fn timed() -> ocl::Result<()> {
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
    let n_tries = 50;
    for i in 1..30 {
        let data_size: usize = 1 << i;

        let ocl_pq = queue.dims(data_size).build()?;

        let mut duration = Duration::new(0, 0);
        for _j in 1..n_tries {
            {
                let kern_start = Instant::now();
                let _buffer: Buffer<u8> = ocl_pq.create_buffer()?;
                duration += kern_start.elapsed();
            }
        }
        let seconds = duration.as_secs_f32();
        let seconds_per_try = seconds / n_tries as f32;
        let time = Time::new::<second>(seconds / n_tries as f32);
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
    match timed() {
        Ok(_) => (),
        Err(err) => println!("{}", err),
    }
}
