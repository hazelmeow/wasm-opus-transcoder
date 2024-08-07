use byteorder::{ByteOrder, LittleEndian};
use ogg::PacketWriteEndInfo;
use rubato::{Resampler, SincFixedIn, SincFixedOut, SincInterpolationParameters};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{cmp::min, error::Error, io::Cursor};
use symphonia::core::{
    audio::{SampleBuffer, SignalSpec},
    codecs::DecoderOptions,
    formats::{FormatOptions, SeekMode, SeekTo},
    io::MediaSourceStream,
    meta::{MetadataOptions, MetadataRevision, StandardTagKey},
    probe::{Hint, ProbeResult},
};
use thiserror::Error;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// Hash bytes with SHA256 and format the result as a hex string.
pub fn hash_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[derive(Debug, Error)]
pub enum TranscodeError {
    #[error("no track")]
    NoTrack,
    #[error("invalid channels")]
    InvalidChannels,
}

#[derive(Debug)]
pub enum Progress {
    Loading,
    Resampling(u8),
    Encoding(u8),
}

#[derive(Debug)]
pub enum ResampleMode {
    Oneshot,
    Chunks,
}

#[derive(Debug)]
pub struct TranscodeOptions {
    resample_mode: ResampleMode,
}

impl Default for TranscodeOptions {
    fn default() -> Self {
        Self {
            resample_mode: ResampleMode::Chunks,
        }
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", wasm_bindgen(getter_with_clone))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TranscodeOutput {
    #[cfg_attr(feature = "serde", serde(with = "serde_bytes"))]
    pub data: Vec<u8>,
    pub metadata: Metadata,
    pub visuals: Vec<ArtOriginal>,
    pub audio_hash: String,
}

pub fn transcode(
    bytes: Vec<u8>,
    options: TranscodeOptions,
    on_progress: impl Fn(Progress),
) -> Result<TranscodeOutput, Box<dyn Error>> {
    on_progress(Progress::Loading);

    let audio_hash = hash_bytes(&bytes);

    let cursor = Cursor::new(bytes);
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

    let hint = Hint::new();

    let mut probe_result = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions {
                enable_gapless: true,
                ..FormatOptions::default()
            },
            &MetadataOptions::default(),
        )
        .unwrap();

    let track = probe_result
        .format
        .default_track()
        .ok_or(TranscodeError::NoTrack)?;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .unwrap();

    // read 1 packet to check the sample rate and channel count
    let spec = {
        let decoded = loop {
            let packet = probe_result.format.next_packet()?;

            match decoder.decode(&packet) {
                Ok(decoded) => break decoded,
                Err(_) => {
                    // failed to decode, keep trying
                    continue;
                }
            }
        };

        *decoded.spec()
    };

    // seek back to start and reset
    probe_result.format.seek(
        SeekMode::Accurate,
        SeekTo::Time {
            time: symphonia::core::units::Time::new(0, 0.0),
            track_id: None,
        },
    )?;
    decoder.reset();

    let (metadata, visuals) = read_metadata(spec, &mut probe_result);

    // read packets
    let mut source = vec![Vec::new(); spec.channels.count()];
    loop {
        match probe_result.format.next_packet() {
            Ok(packet) => {
                let decoded = decoder.decode(&packet)?;

                if decoded.frames() > 0 {
                    let mut sample_buffer: SampleBuffer<f32> =
                        SampleBuffer::new(decoded.frames() as u64, spec);

                    sample_buffer.copy_interleaved_ref(decoded);

                    let samples = sample_buffer.samples();

                    for frame in samples.chunks(spec.channels.count()) {
                        for (chan, sample) in frame.iter().enumerate() {
                            source[chan].push(*sample)
                        }
                    }
                } else {
                    eprintln!("empty packet encountered while decoding");
                }
            }
            Err(symphonia::core::errors::Error::IoError(_)) => {
                // no more packets
                break;
            }
            Err(e) => return Err(e.into()),
        }
    }

    // resample if needed
    let mut resampled = if spec.rate != 48000 {
        match options.resample_mode {
            ResampleMode::Oneshot => resample_oneshot(source, spec, &on_progress)?,
            ResampleMode::Chunks => resample_chunks(source, spec, &on_progress)?,
        }
    } else {
        source
    };

    let samples = if spec.channels.count() == 1 {
        // mono
        resampled.remove(0)
    } else {
        // stereo, interleave for opus
        resampled[0]
            .iter()
            .zip(resampled[1].iter())
            .flat_map(|(a, b)| [a, b])
            .copied()
            .collect()
    };

    let data = encode(samples, spec.channels.count(), &on_progress)?;

    Ok(TranscodeOutput {
        data,
        metadata,
        visuals,
        audio_hash,
    })
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", wasm_bindgen(getter_with_clone))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Metadata {
    /// Duration in seconds
    pub duration: f32,
    pub title: Option<String>,
    pub album: Option<String>,
    pub artist: Option<String>,
    pub track_number: Option<String>,
    pub album_artist: Option<String>,
}

fn read_metadata(spec: SignalSpec, probe_result: &mut ProbeResult) -> (Metadata, Vec<ArtOriginal>) {
    let num_frames = probe_result
        .format
        .default_track()
        .unwrap() // we already know there's at least one track
        .codec_params
        .n_frames
        .unwrap_or(0);

    let mut md = Metadata {
        duration: num_frames as f32 / spec.rate as f32,
        title: None,
        album: None,
        artist: None,
        track_number: None,
        album_artist: None,
    };

    let mut visuals = Vec::new();

    if let Some(rev) = get_metadata_revision(probe_result) {
        let tags = rev.tags();

        for tag in tags {
            match tag.std_key {
                Some(StandardTagKey::TrackTitle) => {
                    md.title = Some(tag.value.to_string());
                }
                Some(StandardTagKey::Album) => {
                    md.album = Some(tag.value.to_string());
                }
                Some(StandardTagKey::Artist) => {
                    md.artist = Some(tag.value.to_string());
                }
                Some(StandardTagKey::TrackNumber) => {
                    md.track_number = Some(tag.value.to_string());
                }
                Some(StandardTagKey::AlbumArtist) => {
                    md.album_artist = Some(tag.value.to_string());
                }

                _ => {}
            }
        }

        for visual in rev.visuals() {
            let hash = hash_bytes(&visual.data);
            visuals.push(ArtOriginal {
                hash,
                media_type: visual.media_type.clone(),
                data: visual.data.clone().into(),
            });
        }
    }

    (md, visuals)
}

fn get_metadata_revision(probe_result: &mut ProbeResult) -> Option<MetadataRevision> {
    if let Some(rev) = probe_result.format.metadata().skip_to_latest() {
        return Some(rev.clone());
    }

    if let Some(rev) = probe_result
        .metadata
        .get()
        .and_then(|mut m| m.skip_to_latest().cloned())
    {
        return Some(rev);
    }

    None
}

fn resample_oneshot(
    source: Vec<Vec<f32>>,
    spec: SignalSpec,
    on_progress: impl Fn(Progress),
) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    on_progress(Progress::Resampling(0));

    let mut resampler = {
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: rubato::SincInterpolationType::Linear,
            oversampling_factor: 128,
            window: rubato::WindowFunction::BlackmanHarris2,
        };
        SincFixedIn::<f32>::new(
            48000_f64 / spec.rate as f64,
            1.0,
            params,
            source[0].len(),
            spec.channels.count(),
        )?
    };

    // resample
    let resampled = resampler.process(&source, None)?;

    on_progress(Progress::Resampling(100));

    Ok(resampled)
}

fn resample_chunks(
    source: Vec<Vec<f32>>,
    spec: SignalSpec,
    on_progress: impl Fn(Progress),
) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    on_progress(Progress::Resampling(0));
    let mut progress = 0;

    let mut resampler = {
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: rubato::SincInterpolationType::Linear,
            oversampling_factor: 128,
            window: rubato::WindowFunction::BlackmanHarris2,
        };
        SincFixedOut::<f32>::new(
            48000_f64 / spec.rate as f64,
            1.0,
            params,
            480,
            spec.channels.count(),
        )?
    };

    let mut resampled = vec![Vec::new(); spec.channels.count()];

    let mut pos = 0;
    while pos < source[0].len() {
        let samples_needed = resampler.input_frames_next();

        let mut chunk_samples: Vec<Vec<f32>> =
            vec![vec![0.0; samples_needed]; spec.channels.count()];

        // copy from source into chunk samples
        // TODO: is this creating some extra samples during the last chunk?
        for i in 0..spec.channels.count() {
            let start = pos;
            let end = min(start + samples_needed, source[0].len());
            chunk_samples[i][..end - start].copy_from_slice(&source[i][start..end]);
        }

        let chunk_resampled = resampler.process(&chunk_samples, None)?;

        // copy resampled chunk to output
        for (channel, samples) in chunk_resampled.into_iter().enumerate() {
            resampled[channel].extend_from_slice(&samples);
        }

        pos += samples_needed;

        let new_progress = if pos >= source[0].len() {
            100
        } else {
            round_progress(pos as f32 / source[0].len() as f32)
        };
        if new_progress != progress {
            progress = new_progress;
            on_progress(Progress::Resampling(progress));
        }
    }

    Ok(resampled)
}

// https://github.com/sheosi/ogg-opus/blob/master/src/encode.rs
fn encode(
    audio: Vec<f32>,
    num_channels: usize,
    on_progress: impl Fn(Progress),
) -> Result<Vec<u8>, Box<dyn Error>> {
    on_progress(Progress::Encoding(0));
    let mut progress = 0;

    const SAMPLE_RATE: usize = 48000; // we resample to 48k first if necessary
    const FRAME_TIME_MS: usize = 20;
    const MAX_PACKET: usize = 4000; // recommended by opus

    let frame_samples = SAMPLE_RATE / 1000 * FRAME_TIME_MS;
    let frame_size = frame_samples * num_channels;

    // TODO: is this chill
    let serial = 0;

    let mut output = Vec::new();
    let mut packet_writer = ogg::PacketWriter::new(&mut output);

    let opus_channels = match num_channels {
        1 => opus::Channels::Mono,
        2 => opus::Channels::Stereo,
        _ => return Err(TranscodeError::InvalidChannels.into()),
    };

    let mut encoder = opus::Encoder::new(48000, opus_channels, opus::Application::Audio)?;
    encoder.set_bitrate(opus::Bitrate::Bits(128000))?;

    #[rustfmt::skip]
	let mut opus_header: [u8; 19] = [
        b'O', b'p', b'u', b's', b'H', b'e', b'a', b'd', // magic signature
        1, // version, always 1
        num_channels as u8, // channel count
        0, 0, // pre-skip, written later
        0, 0, 0, 0, // input sample rate (informational). we don't write this atm
        0, 0, // output gain
        0, // channel mapping family
    ];

    // write pre-skip
    let skip_samples = encoder.get_lookahead().unwrap() as usize;
    LittleEndian::write_u16(&mut opus_header[10..12], skip_samples as u16);

    let mut comment_header = Vec::new();
    comment_header.extend(b"OpusTags"); // magic signature

    // write vendor string
    let vendor = "wasm-opus-transcoder"; // TODO
    let mut len_buf = [0; 4];
    LittleEndian::write_u32(&mut len_buf, vendor.len() as u32);
    comment_header.extend(&len_buf);
    comment_header.extend(vendor.bytes());

    // no metadata
    comment_header.extend([0; 4]);

    // write opus header and comment header packets
    packet_writer.write_packet(&opus_header, serial, ogg::PacketWriteEndInfo::EndPage, 0)?;
    packet_writer.write_packet(&comment_header, serial, ogg::PacketWriteEndInfo::EndPage, 0)?;

    // write all full opus frames
    let total_samples = audio.len() + skip_samples;
    let num_frames = (total_samples as f32 / frame_size as f32).floor() as usize;

    for frame_idx in 0..num_frames {
        let frame_start = frame_idx * frame_size;
        let frame_end = (frame_idx + 1) * frame_size;

        let mut frame_buf = vec![0; MAX_PACKET];

        if frame_start > skip_samples {
            // starts after the padding

            // subtract padding length from frame range when indexing into source samples
            let input = &audio[frame_start - skip_samples..frame_end - skip_samples];

            // encode
            let packet_len = encoder.encode_float(input, &mut frame_buf)?;
            frame_buf.truncate(packet_len);
        } else {
            // starts before the padding

            // fill with 0s
            let mut input = vec![0.0; frame_size];

            // if the frame ends outside the padding
            if frame_end > skip_samples {
                // copy into input, skipping `skip_samples - frame_start` 0s
                input[skip_samples - frame_start..]
                    .copy_from_slice(&audio[..frame_end - skip_samples]);
            }

            // encode
            let packet_len = encoder.encode_float(&input, &mut frame_buf)?;
            frame_buf.truncate(packet_len);
        }

        let end_info = if frame_end == total_samples {
            PacketWriteEndInfo::EndStream
        } else {
            PacketWriteEndInfo::NormalPacket
        };

        let granule_position = (frame_idx + 1) * frame_samples;

        packet_writer.write_packet(frame_buf, serial, end_info, granule_position as u64)?;

        let new_progress = round_progress((frame_idx + 1) as f32 / (num_frames + 1) as f32);
        if new_progress != progress {
            progress = new_progress;
            on_progress(Progress::Encoding(progress));
        }
    }

    // write final frame
    {
        // start after last full frame
        let frame_start = num_frames * frame_size;

        // number of 0 samples in final frame
        let skip = skip_samples - min(frame_start, skip_samples);

        // number of source samples in final frame
        let rem_samples = total_samples - frame_start;

        // where to start audio samples
        let audio_start = if frame_start <= skip_samples {
            0
        } else {
            frame_start - skip_samples
        };

        let mut input = vec![0.0; frame_size];
        input[skip..rem_samples].copy_from_slice(&audio[audio_start..]);

        let mut frame_buf = vec![0; MAX_PACKET];

        let packet_len = encoder.encode_float(&input, &mut frame_buf)?;
        frame_buf.truncate(packet_len);

        let granule_position = total_samples / num_channels;

        packet_writer.write_packet(
            frame_buf,
            serial,
            PacketWriteEndInfo::EndStream,
            granule_position as u64,
        )?;
    }

    on_progress(Progress::Encoding(100));

    Ok(output)
}

fn round_progress(p: f32) -> u8 {
    (p * 100.0).trunc() as u8
}

#[derive(Debug, Error)]
pub enum ArtError {
    #[error("unsupported format: {0}")]
    MediaType(String),
    #[error("decoding failed: {0}")]
    Decode(String),
    #[error("encoding failed: {0}")]
    Encode(String),
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", wasm_bindgen(getter_with_clone))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ArtOriginal {
    pub hash: String,
    pub media_type: String,
    #[cfg_attr(feature = "serde", serde(with = "serde_bytes"))]
    pub data: Vec<u8>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", wasm_bindgen(getter_with_clone))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Art {
    pub hash: String,
    #[cfg_attr(feature = "serde", serde(with = "serde_bytes"))]
    pub data_512: Vec<u8>,
    #[cfg_attr(feature = "serde", serde(with = "serde_bytes"))]
    pub data_2048: Vec<u8>,
}

pub fn process_art(media_type: &str, bytes: Vec<u8>) -> Result<Art, Box<dyn Error>> {
    // for bundle size, we only support png and jpg visuals
    if matches!(media_type, "image/png" | "image/jpg" | "image/jpeg") {
        let reader = image::io::Reader::new(Cursor::new(&bytes))
            .with_guessed_format()
            .expect("cursor io never fails");

        let image = reader
            .decode()
            .map_err(|e| ArtError::Decode(e.to_string()))?;

        let resized_512 = image.resize(512, 512, image::imageops::FilterType::Nearest);
        let mut data_512: Vec<u8> = vec![];
        let mut encoder_512 = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut data_512, 90);
        encoder_512
            .encode_image(&resized_512)
            .map_err(|e| ArtError::Encode(e.to_string()))?;

        let resized_2048 = image.resize(2048, 2048, image::imageops::FilterType::Nearest);
        let mut data_2048: Vec<u8> = vec![];
        let mut encoder_2048 =
            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut data_2048, 95);
        encoder_2048
            .encode_image(&resized_2048)
            .map_err(|e| ArtError::Encode(e.to_string()))?;

        // compute hash of input
        let hash = hash_bytes(&bytes);

        Ok(Art {
            hash,
            data_512,
            data_2048,
        })
    } else {
        Err(ArtError::MediaType(media_type.to_owned()).into())
    }
}
