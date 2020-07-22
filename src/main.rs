use std::time::{Duration, Instant};
use structopt::StructOpt;

use prettytable::{cell, row, Table, format};

use pbr::ProgressBar;

/// Configuration struct gpu benchmarking
#[derive(StructOpt, Debug)]
#[structopt(author, about)]
struct Configuration {
    /// At which power of two to stop the test
    #[structopt(long, short = "n", default_value = "25")]
    end_power: usize,

    /// The number of iterations per data-size
    #[structopt(long, short = "t", default_value = "50")]
    tries: usize,

    /// Whether to verify the data of the copy. Can take a long time.
    #[structopt(long, short = "v")]
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
    (2..=max_power)
        .map(|power| (2.0 as f32).powi(power as i32) as usize)
        .collect()
}

fn get_min_max_avg(values: Vec<Duration>) -> (f32, f32, f32) {
    let sum = values.iter().sum::<Duration>().as_secs_f32() * 1000.0;
    let min = values
        .iter()
        .min()
        .unwrap_or(&Duration::from_secs(0))
        .as_secs_f32() * 1000.0;
    let max = values
        .iter()
        .max()
        .unwrap_or(&Duration::from_secs(0))
        .as_secs_f32() * 1000.0;
    (min as f32, max as f32, sum as f32 / values.len() as f32)
}

fn create_tables() -> (Table, Table) {
    let mut tables = (Table::new(), Table::new());
    tables.0.set_format(*format::consts::FORMAT_BOX_CHARS);
    tables.1.set_format(*format::consts::FORMAT_BOX_CHARS);
    tables.0.add_row(row![
        "Iteration",
        "Datasize (bytes)",
        "Datasize (MB)",
        "min Time (ms)",
        "max (ms)",
        "avg Time (ms)",
        "Bandwidth (MB/s)"
    ]);
    tables.1.add_row(row![
        "Iteration",
        "Datasize (bytes)",
        "Datasize (MB)",
        "min Time (ms)",
        "max (ms)",
        "avg Time (ms)",
        "Bandwidth (MB/s)"
    ]);
    tables
}

fn add_measurement(table: &mut Table, iteration: usize, data_size: usize, timings: Vec<Duration>) {
    let (min, max, avg) = get_min_max_avg(timings);
    let data_size_mb = data_size as f32 / 1024.0 / 1024.0;
    let bandwidth = data_size_mb / avg * 1000.0;
    table.add_row(row![iteration, data_size, format!("{:.2}", data_size_mb), format!("{:.2}", min), format!("{:.2}", max), format!("{:.2}", avg), format!("{:.2}", bandwidth)]);
}

async fn run(config: Configuration) {
    let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::Default,
            compatible_surface: None,
        })
        .await
        .unwrap();

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
                shader_validation: true,
            },
            None,
        )
        .await
        .unwrap();

    let mut tables = create_tables();

    let data_sizes = get_default_sizes();
    //let data_sizes = get_power_two_sizes(config.end_power as u32);

    println!("Running {} tests...", data_sizes.len());
    let mut pb = ProgressBar::new(data_sizes.len() as u64);
    pb.format("╢▌▌░╟");

    for (iteration, data_size) in data_sizes.iter().enumerate() {
        let mut upload_data = vec![iteration as u8; *data_size];
        let mut download_data = vec![0 as u8; *data_size];

        let mut upload_times = Vec::with_capacity(config.tries);
        let mut download_times = Vec::with_capacity(config.tries);
        for _ in 1..=config.tries {
            let expected_sum = iteration * data_size;
            let (upload_time, download_time) = execute_gpu(
                &device,
                &queue,
                expected_sum,
                &mut upload_data,
                &mut download_data,
                config.verify,
            )
            .await;
            upload_times.push(upload_time);
            download_times.push(download_time);
        }
        add_measurement(&mut tables.0, iteration, *data_size, upload_times);
        add_measurement(&mut tables.1, iteration, *data_size, download_times);

        pb.inc();
    }
    pb.finish_print("Finished test");
    println!("Upload times");
    tables.0.printstd();

    println!("Download times");
    tables.1.printstd();
}

async fn execute_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    expected_sum: usize,
    host_data_upload: &[u8],
    host_data_download: &mut [u8],
    verify: bool,
) -> (Duration, Duration) {
    let slice_size = host_data_upload.len() * std::mem::size_of::<u8>();
    let size = slice_size as wgpu::BufferAddress;

    let upload_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsage::MAP_WRITE | wgpu::BufferUsage::COPY_SRC,
        mapped_at_creation: false,
    });

    let download_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        size,
        usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::MAP_READ,
        label: None,
        mapped_at_creation: false,
    });
    device.poll(wgpu::Maintain::Wait);

    let upload_time = {
        let start = std::time::Instant::now();
        let buffer_slice = upload_buffer.slice(..);
        let buffer_future = buffer_slice.map_async(wgpu::MapMode::Write);
        device.poll(wgpu::Maintain::Wait);

        if let Ok(_) = buffer_future.await {
            let mut data = buffer_slice.get_mapped_range_mut();
            data.copy_from_slice(host_data_upload);
            device.poll(wgpu::Maintain::Wait);
            drop(data);
            upload_buffer.unmap();
        } else {
            println!("oops");
        }
        
        start.elapsed()
    };

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(&upload_buffer, 0, &download_buffer, 0, size);
    queue.submit(Some(encoder.finish()));

    device.poll(wgpu::Maintain::Wait);

    let download_time = {
        let start = std::time::Instant::now();
        let mut end_time = Duration::from_secs(0);

        let buffer_slice = download_buffer.slice(..);
        let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);
        device.poll(wgpu::Maintain::Wait);

        if let Ok(_) = buffer_future.await {
            let data = buffer_slice.get_mapped_range();
            host_data_download.copy_from_slice(&data);
            drop(data);
            download_buffer.unmap();
            end_time = start.elapsed();

            if verify {
                let mut total: usize = 0;
                for item in host_data_download {
                    total += *item as usize;
                }
                assert!(total == expected_sum);
            }
        } else {
            println!("oops");
        }

        end_time
    };
    (upload_time, download_time)
}

pub fn main() {
    let config = Configuration::from_args();
    println!("{:?}", config);
    futures::executor::block_on(run(config));
}
