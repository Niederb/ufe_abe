use std::time::{Duration, Instant};

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
        "Datasize (bytes)",
        "Time (ms)",
        "Bandwidth (MB/s)"
    ]);

    let mut pb = ProgressBar::new((config.end_power + 1) as u64);
    pb.format("╢▌▌░╟");

    let number = 7;
    for i in 0..30 {
        let data_size = (2.0 as f32).powi(i as i32) as usize;
        let numbers = vec![number; data_size];
        // To see the output, run `RUST_LOG=info cargo run --example hello-compute`.
        
        let end_time = execute_gpu(&device, &queue, numbers).await;
        let bandwidth = data_size as f32 / 1024.0 / 1024.0 / end_time.as_millis() as f32 * 1000.0;
        table.add_row(row![
            i,
            data_size / 1024 / 1024,
            end_time.as_millis(),
            bandwidth
        ]);
        pb.inc();
    }
    pb.finish_print("Finished test");
    table.printstd();
}

async fn execute_gpu(device: &wgpu::Device, queue: &wgpu::Queue, numbers: Vec<u32>) -> Duration {
    let slice_size = numbers.len() * std::mem::size_of::<u32>();
    let size = slice_size as wgpu::BufferAddress;
    
    let staging_buffer = device.create_buffer_with_data(
        bytemuck::cast_slice(&numbers),
        wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::COPY_SRC,
    );

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
    let buffer_future = staging_buffer.map_read(0, size);

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    let start = std::time::Instant::now();
    device.poll(wgpu::Maintain::Wait);
    let end_time = start.elapsed();

    if let Ok(mapping) = buffer_future.await {
        /*mapping
            .as_slice()
            .chunks_exact(4)
            .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
            .collect()*/
    } else {
        panic!("failed to run compute on gpu!")
    }
    end_time
}

pub fn main() {
    //env_logger::init();
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
    let end_power = value_t!(matches, "end-power", usize).unwrap_or(30);

    let config = Configuration {
        end_power,
        tries,
    };
    println!("Number of tries per test: {}", config.tries);
    println!("Running {} tests...", (config.end_power + 1));

    futures::executor::block_on(run(config));
/*
    match timed(config) {
        Ok(_) => (),
        Err(err) => println!("{}", err),
    }*/
}
