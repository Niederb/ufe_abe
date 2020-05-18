use std::time::Duration;
use structopt::StructOpt;

use prettytable::{cell, row, Table};

use pbr::ProgressBar;

/// Configuration struct gpu benchmarking
#[derive(StructOpt, Debug)]
#[structopt(author, about)]
struct Configuration {
    /// At which power of two to stop the test
    #[structopt(long, short = "n", default_value = "25")]
    end_power: usize,

    /// The number of iterations per data-size
    #[structopt(long, short = "t", default_value = "25")]
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
    (0..=max_power)
        .map(|power| (2.0 as f32).powi(power as i32) as usize)
        .collect()
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
        let mut upload_data = vec![iteration as u8; *data_size];
        let mut download_data = vec![0 as u8; *data_size];

        let start_time = std::time::Instant::now();
        let mut total_time = Duration::new(0, 0);
        let mut max_time = Duration::new(0, 0);
        let mut min_time = Duration::new(std::u64::MAX, 0);
        for _ in 1..=config.tries {
            let expected_sum = iteration * data_size;
            let (upload_time, download_time) = execute_gpu(
                expected_sum,
                &device,
                &queue,
                &mut upload_data,
                &mut download_data,
            )
            .await;
            let end_time = download_time;
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

async fn execute_gpu(
    expected_sum: usize,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    host_data_upload: &[u8],
    host_data_download: &mut [u8],
) -> (Duration, Duration) {
    let slice_size = host_data_upload.len() * std::mem::size_of::<u8>();
    let size = slice_size as wgpu::BufferAddress;

    let upload_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        size,
        usage: wgpu::BufferUsage::STORAGE
            | wgpu::BufferUsage::COPY_DST
            | wgpu::BufferUsage::COPY_SRC
            | wgpu::BufferUsage::MAP_READ
            | wgpu::BufferUsage::MAP_WRITE,
        label: None,
    });
    device.poll(wgpu::Maintain::Wait);

    let upload_time = {
        let start = std::time::Instant::now();
        let write_result = upload_buffer.map_write(0, size);

        device.poll(wgpu::Maintain::Wait);

        if let Ok(mut mapping) = write_result.await {
            device.poll(wgpu::Maintain::Wait);
            mapping.as_slice().copy_from_slice(host_data_upload);
        }
        device.poll(wgpu::Maintain::Wait);
        start.elapsed()
    };
    let download_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        size,
        usage: wgpu::BufferUsage::STORAGE
            | wgpu::BufferUsage::COPY_DST
            | wgpu::BufferUsage::COPY_SRC
            | wgpu::BufferUsage::MAP_READ
            | wgpu::BufferUsage::MAP_WRITE,
        label: None,
    });
    device.poll(wgpu::Maintain::Wait);

    /*let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(&upload_buffer, 0, &download_buffer, 0, size);*/
    //encoder.copy_buffer_to_buffer(&download_buffer, 0, &upload_buffer, 0, size);

    //queue.submit(&[encoder.finish()]);
    device.poll(wgpu::Maintain::Wait);

    let download_time = {
        let start = std::time::Instant::now();
        let buffer_future = download_buffer.map_read(0, size);
        device.poll(wgpu::Maintain::Wait);

        let result = buffer_future.await;
        let mut end_time = start.elapsed();
        if let Ok(mapping) = result {
            host_data_download.copy_from_slice(mapping.as_slice());
            end_time = start.elapsed();
            //mapping.as_slice().copy_from_slice(download_data);
            //download_data.copy_from_slice(host_data);
            /*unsafe {
                //std::ptr::copy_nonoverlapping(download_data.as_ptr(), host_data.as_mut_ptr(), size as usize);
                std::ptr::copy_nonoverlapping(host_data.as_ptr(), download_data.as_mut_ptr(), size as usize);
            }*/
            //host_data.copy_from_slice(download_data);
            /*let mut total: usize = 0;
            for item in mapping.as_slice() {
                total += *item as usize;
            }
            assert!(total == expected_sum);*/
        }
        device.poll(wgpu::Maintain::Wait);
        end_time
    };
    (upload_time, download_time)
}

pub fn main() {
    let config = Configuration::from_args();
    println!("{:?}", config);
    futures::executor::block_on(run(config));
}
