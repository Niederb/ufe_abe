use std::time::Duration;

#[macro_use]
extern crate clap;
use clap::{App, Arg};

#[macro_use]
extern crate prettytable;
use prettytable::Table;

extern crate pbr;
use pbr::ProgressBar;

use simplelog::*;
use std::fs::File;

struct Configuration {
    end_power: usize,
    tries: usize,
}

async fn run(config: Configuration) {
    let adapter = wgpu::Adapter::request(
        &wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::Default,
            compatible_surface: None,
        },
        wgpu::BackendBit::PRIMARY,
    )
    .await
    .unwrap();

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            extensions: wgpu::Extensions {
                anisotropic_filtering: false,
            },
            limits: wgpu::Limits::default(),
        })
        .await;

    let mut table = Table::new();
    table.add_row(row![
        "Iteration",
        "Datasize (MB)",
        "min Time (ms)",
        "max (ms)",
        "avg Time (ms)",
        "Bandwidth (MB/s)"
    ]);

    let mut pb = ProgressBar::new((config.end_power + 1) as u64);
    pb.format("╢▌▌░╟");

    let number = 7;
    for i in 0..=config.end_power {
        let data_size = (2.0 as f32).powi(i as i32) as usize;
        let numbers = vec![number as u8; data_size];
        // To see the output, run `RUST_LOG=info cargo run --example hello-compute`.
        let start_time = std::time::Instant::now();
        let mut total_time = Duration::new(0, 0);
        let mut max_time = Duration::new(0, 0);
        let mut min_time = Duration::new(std::u64::MAX, 0);
        for _ in 1..=config.tries {
            let end_time = execute_gpu(&device, &queue, &numbers).await;
            total_time += end_time;
            max_time = if end_time > max_time {
                end_time
            } else {
                max_time
            };
            min_time = if end_time < min_time {
                end_time
            } else {
                min_time
            };
            //std::thread::sleep(Duration::from_millis(10));
        }
        //let total_time = start_time.elapsed();
        let data_size = data_size as f32 / 1024.0 / 1024.0;
        let avg_time_millis = total_time.as_millis() as f32 / config.tries as f32;
        let bandwidth = data_size / avg_time_millis * 1000.0;
        table.add_row(row![
            i,
            data_size,
            min_time.as_millis(),
            max_time.as_millis(),
            avg_time_millis,
            bandwidth
        ]);
        pb.inc();
    }
    pb.finish_print("Finished test");
    table.printstd();
}

async fn execute_gpu(device: &wgpu::Device, queue: &wgpu::Queue, numbers: &[u8]) -> Duration {
    let slice_size = numbers.len() * std::mem::size_of::<u8>();
    let size = slice_size as wgpu::BufferAddress;

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        size,
        usage: wgpu::BufferUsage::STORAGE
            | wgpu::BufferUsage::COPY_DST
            | wgpu::BufferUsage::COPY_SRC
            | wgpu::BufferUsage::READ_ALL
            | wgpu::BufferUsage::WRITE_ALL,
        label: None,
    });
    device.poll(wgpu::Maintain::Wait);
    
    let end_time = {    
        let start = std::time::Instant::now();
        let write_result = staging_buffer.map_write(
            0,
            size
        );
        
        device.poll(wgpu::Maintain::Wait);
        
        if let Ok(mut mapping) = write_result.await {
            
            mapping.as_slice().copy_from_slice(numbers);
        }
        device.poll(wgpu::Maintain::Wait);
        start.elapsed()
    };
    /*let staging_buffer = device.create_buffer_with_data(
        bytemuck::cast_slice(&numbers),
        wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::COPY_SRC,
    );*/
    

    let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        size,
        usage: wgpu::BufferUsage::STORAGE
            | wgpu::BufferUsage::COPY_DST
            | wgpu::BufferUsage::COPY_SRC,
        label: None,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(&staging_buffer, 0, &storage_buffer, 0, size);

    encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, size);

    queue.submit(&[encoder.finish()]);

    // Note that we're not calling `.await` here.
    //let buffer_future = staging_buffer.map_read(0, size);

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.

    device.poll(wgpu::Maintain::Wait);

    /*let result = buffer_future.await;

    if let Ok(mapping) = result {
        /*mapping
        .as_slice()
        .chunks_exact(4)
        .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
        .collect()*/
    } else {
        panic!("failed to run compute on gpu!")
    }*/
    //drop(staging_buffer);
    //drop(storage_buffer);

    end_time
}

pub fn main() {
    CombinedLogger::init(vec![
        TermLogger::new(LevelFilter::Error, Config::default(), TerminalMode::Mixed).unwrap(),
        WriteLogger::new(
            LevelFilter::Error,
            Config::default(),
            File::create("log_file.txt").unwrap(),
        ),
    ])
    .unwrap();
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
        .arg(
            Arg::with_name("pinned")
                .short("p")
                .help("Use pinned host memory"),
        )
        .get_matches();

    let tries = value_t!(matches, "tries", usize).unwrap_or(50);
    let end_power = value_t!(matches, "end-power", usize).unwrap_or(28);

    let config = Configuration { end_power, tries };
    println!("Number of tries per test: {}", config.tries);
    println!("Running {} tests...", (config.end_power + 1));

    futures::executor::block_on(run(config));
    /*
    match timed(config) {
        Ok(_) => (),
        Err(err) => println!("{}", err),
    }*/
}
