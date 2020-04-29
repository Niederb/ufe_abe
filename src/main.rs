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
    verify: bool,
}

fn get_default_sizes() -> Vec<usize> {
    let mut sizes = Vec::new();
    let mut next_size = 1024;
    while next_size <= 67_186_688 {
        sizes.push(next_size);
        if next_size < 20_480 {
            next_size += 1_024;
        } else if next_size < 51_200 {
            next_size += 2_048;
        } else if next_size < 102_400 {
            next_size += 10_240;
        } else if next_size < 1126400 {
            next_size += 102_400;
        } else if next_size < 16_855_040 {
            next_size += 1_048_576;
        } else if next_size < 33_632_256 {
            next_size += 2_097_152;
        } else {
            next_size += 4_194_304;
        }
    }

    sizes
}

fn get_power_two_sizes(max_power: u32) -> Vec<usize> {
    (0..=max_power).map(| power| (2.0 as f32).powi(power as i32) as usize).collect()
    //let data_size = (2.0 as f32).powi(i as i32) as usize;
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

    let data_sizes = get_default_sizes();
    //let data_sizes = get_power_two_sizes(config.end_power);
    
    println!("Running {} tests...", data_sizes.len());
    let mut pb = ProgressBar::new(data_sizes.len() as u64);
    pb.format("╢▌▌░╟");

    for (iteration, data_size) in data_sizes.iter().enumerate() {
        let numbers = vec![iteration as u8; *data_size];

        let start_time = std::time::Instant::now();
        let mut total_time = Duration::new(0, 0);
        let mut max_time = Duration::new(0, 0);
        let mut min_time = Duration::new(std::u64::MAX, 0);
        for _ in 1..=config.tries {
            let expected_sum = iteration * data_size;
            let end_time = execute_gpu(expected_sum, &device, &queue, &numbers).await;
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
        }
        //let total_time = start_time.elapsed();
        let data_size = *data_size as f32 / 1024.0 / 1024.0;
        let avg_time_millis = total_time.as_millis() as f32 / config.tries as f32;
        let bandwidth = data_size / avg_time_millis * 1000.0;
        table.add_row(row![
            iteration,
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

async fn execute_gpu(expected_sum: usize, device: &wgpu::Device, queue: &wgpu::Queue, host_data: &[u8]) -> Duration {
    let slice_size = host_data.len() * std::mem::size_of::<u8>();
    let size = slice_size as wgpu::BufferAddress;

    let upload_buffer = device.create_buffer(&wgpu::BufferDescriptor {
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
        let write_result = upload_buffer.map_write(
            0,
            size
        );
        
        device.poll(wgpu::Maintain::Wait);
        
        if let Ok(mut mapping) = write_result.await {
            mapping.as_slice().copy_from_slice(host_data);
        }
        device.poll(wgpu::Maintain::Wait);
        start.elapsed()
    };
    /*let upload_buffer = device.create_buffer_with_data(
        bytemuck::cast_slice(&numbers),
        wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::COPY_SRC,
    );*/
    

    let download_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        size,
        usage: wgpu::BufferUsage::STORAGE
            | wgpu::BufferUsage::COPY_DST
            | wgpu::BufferUsage::COPY_SRC
            | wgpu::BufferUsage::READ_ALL,
        label: None,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(&upload_buffer, 0, &download_buffer, 0, size);

    //encoder.copy_buffer_to_buffer(&download_buffer, 0, &upload_buffer, 0, size);

    queue.submit(&[encoder.finish()]);

    let buffer_future = download_buffer.map_read(0, size);

    device.poll(wgpu::Maintain::Wait);

    let result = buffer_future.await;

    if let Ok(mapping) = result {
        let mut total: usize = 0;
        for item in mapping.as_slice() {
            total += *item as usize;
        }
        println!("{}/{}", total, expected_sum);
        assert!(total == expected_sum);
    } else {
        panic!("failed to run compute on gpu!");
    }
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
    ]).unwrap();

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

    let config = Configuration { end_power, tries, verify: true };
    println!("Number of tries per test: {}", config.tries);

    futures::executor::block_on(run(config));
}
