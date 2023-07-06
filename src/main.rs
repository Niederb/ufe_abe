use std::mem;
use std::time::{Duration, Instant};
use structopt::StructOpt;

use prettytable::{cell, format, row, Table};

use pbr::ProgressBar;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TimestampData {
    start: u64,
    end: u64,
}

/// Configuration struct gpu benchmarking
#[derive(StructOpt, Debug)]
#[structopt(author, about)]
struct Configuration {
    /// At which power of two to stop the test
    #[structopt(long, short = "n", default_value = "25")]
    end_power: usize,

    /// The number of iterations per data-size
    #[structopt(long, short = "t", default_value = "50")]
    tries: u32,

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
        .map(|power| 2.0_f32.powi(power as i32) as usize)
        .collect()
}

fn get_min_max_avg(values: &[Duration]) -> (f32, f32, f32) {
    let sum = values.iter().sum::<Duration>().as_secs_f32() * 1000.0;
    let min = values
        .iter()
        .min()
        .unwrap_or(&Duration::from_secs(0))
        .as_secs_f32()
        * 1000.0;
    let max = values
        .iter()
        .max()
        .unwrap_or(&Duration::from_secs(0))
        .as_secs_f32()
        * 1000.0;
    (min as f32, max as f32, sum as f32 / values.len() as f32)
}

fn create_tables() -> Vec<Table> {
    let mut tables = vec![Table::new(); 3];
    for t in tables.iter_mut() {
        t.set_format(*format::consts::FORMAT_BOX_CHARS);
        t.add_row(row![
            "Iteration",
            "Datasize (bytes)",
            "Datasize (MB)",
            "min Time (ms)",
            "max (ms)",
            "avg Time (ms)",
            "Bandwidth (MB/s)"
        ]);
    }
    tables
}

fn add_measurement(table: &mut Table, iteration: usize, data_size: usize, timings: &[Duration]) {
    let (min, max, avg) = get_min_max_avg(timings);
    let data_size_mb = data_size as f32 / 1024.0 / 1024.0;
    let bandwidth = data_size_mb / avg * 1000.0;
    table.add_row(row![
        iteration,
        data_size,
        format!("{:.2}", data_size_mb),
        format!("{:.2}", min),
        format!("{:.2}", max),
        format!("{:.2}", avg),
        format!("{:.2}", bandwidth)
    ]);
}

async fn run(config: Configuration) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        dx12_shader_compiler: wgpu::Dx12Compiler::Fxc,
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .unwrap();
    println!("using adapter: {:?}", adapter);

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::TIMESTAMP_QUERY
                    | wgpu::Features::PIPELINE_STATISTICS_QUERY,
                limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .unwrap();

    let mut tables = create_tables();
    let timestamp_period = queue.get_timestamp_period();

    let data_sizes = get_default_sizes();
    //let data_sizes = get_power_two_sizes(config.end_power as u32);

    println!("Running {} tests...", data_sizes.len());
    let mut pb = ProgressBar::new(data_sizes.len() as u64);
    pb.format("╢▌▌░╟");

    for (iteration, data_size) in data_sizes.iter().enumerate() {
        let upload_data = vec![iteration as u8; *data_size];
        let mut download_data = vec![0; *data_size];

        let mut times = vec![Vec::with_capacity(config.tries as usize); 3];

        for _ in 1..=config.tries {
            let expected_sum = iteration * data_size;
            let timings = execute_gpu(
                &device,
                &queue,
                expected_sum,
                &upload_data,
                &mut download_data,
                config.verify,
            )
            .await;

            for it in times.iter_mut().zip(timings.iter()) {
                let (times, timing) = it;
                times.push(*timing);
            }
        }
        for it in tables.iter_mut().zip(times.iter()) {
            let (table, times) = it;
            add_measurement(table, iteration, *data_size, &times[..]);
        }

        pb.inc();
    }
    pb.finish_print("Finished test");
    println!("Upload times");
    tables[0].printstd();

    println!("GPU/GPU transfer times");
    tables[1].printstd();

    println!("Download times");
    tables[2].printstd();
}

async fn execute_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    expected_sum: usize,
    host_data_upload: &[u8],
    host_data_download: &mut [u8],
    verify: bool,
) -> Vec<Duration> {
    let slice_size = host_data_upload.len() * std::mem::size_of::<u8>();
    let size = slice_size as wgpu::BufferAddress;

    let upload_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let download_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        label: None,
        mapped_at_creation: false,
    });
    device.poll(wgpu::Maintain::Wait);

    let upload_time = {
        let start = Instant::now();
        let buffer_slice = upload_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Write, move |v| sender.send(v).unwrap());
        device.poll(wgpu::Maintain::Wait);

        if let Some(Ok(())) = receiver.receive().await {
            let mut data = buffer_slice.get_mapped_range_mut();
            data.copy_from_slice(host_data_upload);
            device.poll(wgpu::Maintain::Wait);
            drop(data);
            upload_buffer.unmap();
        } else {
            println!("oops");
        }
        device.poll(wgpu::Maintain::Wait);
        start.elapsed()
    };

    let timing_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("timing buffer"),
        size: 2 * mem::size_of::<u64>() as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: None,
        count: 2,
        ty: wgpu::QueryType::Timestamp,
    });
    // GPU/GPU transfer
    let gpu_gpu_time = {
        let start = Instant::now();
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.write_timestamp(&query_set, 0);
        encoder.copy_buffer_to_buffer(&upload_buffer, 0, &download_buffer, 0, size);
        encoder.write_timestamp(&query_set, 1);
        encoder.resolve_query_set(&query_set, 0..2, &timing_buffer, 0);
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        let _ = timing_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        // Wait for device to be done rendering mipmaps
        device.poll(wgpu::Maintain::Wait);
        if let Some(Ok(())) = receiver.receive().await {
            let view = timing_buffer.slice(..).get_mapped_range();
            // Convert the raw data into a useful structure
            let data: &TimestampData = bytemuck::from_bytes(&*view);
            //println!("sdf: {} us", (data.end - data.start)/1000);
            Duration::from_nanos(data.end - data.start)
            //start.elapsed()
        } else {
            Duration::default()
        }
    };

    let download_time = {
        let start = Instant::now();
        let mut end_time = Duration::from_secs(0);

        let buffer_slice = download_buffer.slice(..);

        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        device.poll(wgpu::Maintain::Wait);

        if let Some(Ok(())) = receiver.receive().await {
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
                assert_eq!(expected_sum, total);
            }
        } else {
            println!("oops");
        }
        device.poll(wgpu::Maintain::Wait);
        end_time
    };
    vec![upload_time, gpu_gpu_time, download_time]
}

pub fn main() {
    let config = Configuration::from_args();
    println!("{:?}", config);
    futures::executor::block_on(run(config));
}
