use std::time::{Duration, Instant};

use ocl::flags::{CommandQueueProperties, MemFlags};
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

use std::{convert::TryInto, str::FromStr};
use simplelog::*;
use std::fs::File;

struct Configuration {
    end_power: usize,
    tries: usize,
    memory_flags: MemFlags,
}


async fn run(config: Configuration) {
    let mut table = Table::new();
    table.add_row(row![
        "Iteration",
        "Datasize (bytes)",
        "Time (ms)"
    ]);

    let mut pb = ProgressBar::new((config.end_power + 1) as u64);
    pb.format("╢▌▌░╟");

    let number = 7;
    for i in 0..30 {
        let data_size = (2.0 as f32).powi(i as i32) as usize;
        let numbers = vec![number; data_size];
        // To see the output, run `RUST_LOG=info cargo run --example hello-compute`.
        
        let end_time = execute_gpu(numbers).await;
        //log::error!("Times: {:?}", result);
        //log::error!("Iteration: {}", i);

        table.add_row(row![
            i,
            data_size,
            end_time.as_millis()
        ]);
        pb.inc();
    }
    pb.finish_print("Finished test");
    table.printstd();
}

async fn execute_gpu(numbers: Vec<u32>) -> Duration {
    let slice_size = numbers.len() * std::mem::size_of::<u32>();
    let size = slice_size as wgpu::BufferAddress;

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

    /*let cs = include_bytes!("shader.comp.spv");
    let cs_module =
        device.create_shader_module(&wgpu::read_spirv(std::io::Cursor::new(&cs[..])).unwrap());*/

    let start = std::time::Instant::now();
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

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        bindings: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStage::COMPUTE,
            ty: wgpu::BindingType::StorageBuffer {
                dynamic: false,
                readonly: false,
            },
        }],
        label: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        bindings: &[wgpu::Binding {
            binding: 0,
            resource: wgpu::BindingResource::Buffer {
                buffer: &storage_buffer,
                range: 0..size,
            },
        }],
        label: None,
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        bind_group_layouts: &[&bind_group_layout],
    });

    /*let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        layout: &pipeline_layout,
        compute_stage: wgpu::ProgrammableStageDescriptor {
            module: &cs_module,
            entry_point: "main",
        },
    });*/

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(&staging_buffer, 0, &storage_buffer, 0, size);
    {
        /*let mut cpass = encoder.begin_compute_pass();
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch(numbers.len() as u32, 1, 1);*/
    }
    encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, size);

    queue.submit(&[encoder.finish()]);

    // Note that we're not calling `.await` here.
    let buffer_future = staging_buffer.map_read(0, size);

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    device.poll(wgpu::Maintain::Wait);
    let end_time = start.elapsed();
    //log::error!("Times: {:?}", result);
    let data_size = numbers.len() * 8 / 1024 / 1024;
    //let bandwidth = data_size / end.in_millis();
    //log::error!("Size: {} Time: {:?}", data_size, end);

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

fn timed(config: Configuration) -> ocl::Result<()> {
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
    let mut pb = ProgressBar::new((config.end_power + 1) as u64);
    pb.format("╢▌▌░╟");

    for i in 0..=config.end_power {
        let data_size: usize = 1 << i;
        let ocl_pq = queue.dims(data_size).build()?;

        let mut allocation_duration = Duration::new(0, 0);
        let mut upload_duration = Duration::new(0, 0);
        let mut download_duration = Duration::new(0, 0);
        let mut host_memory = vec![0u8; data_size];
        for j in 0..config.tries {
            {
                // let mut buffer: Buffer<u8> = ocl_pq.create_buffer()?;
                let buffer: Buffer<u8> = ocl_pq
                    .buffer_builder()
                    .flags(config.memory_flags)
                    .fill_val(0)
                    .build()?;
                let start = Instant::now();
                //buffer.write(&host_memory).enq()?;
                buffer.read(&mut host_memory).enq()?;
                allocation_duration += start.elapsed();
            }
        }
        let seconds = allocation_duration.as_secs_f32();
        let seconds_per_try = seconds / config.tries as f32;
        let time = Time::new::<second>(seconds / config.tries as f32);
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

    let memory_flags = if matches.is_present("pinned") {
        MemFlags::new()
            .alloc_host_ptr()
            .write_only()
            .host_read_only()
    } else {
        MemFlags::new()
    };

    let config = Configuration {
        end_power,
        tries,
        memory_flags,
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
