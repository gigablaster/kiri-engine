// Copyright (C) 2024-2025 gigablaster

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

use async_trait::async_trait;
use bytes::Bytes;
use core::slice;
use kiri_common::Align;
use memmap2::{Mmap, MmapOptions};
use speedy::{Readable, Writable};
use std::{
    collections::HashMap,
    fs::File,
    io::{self, Cursor, Read, Seek, Write},
    mem,
    path::Path,
};

use crate::{Archive, ArchiveLoad, AssetReference};

#[derive(Debug, Readable, Writable)]
struct AssetHeader {
    pub offset: u64,
    pub size: u64,
    pub packed: Option<u64>,
}

#[derive(Debug, Default, Readable, Writable)]
struct Directory {
    pub assets: HashMap<AssetReference, AssetHeader>,
}

impl Directory {
    pub fn load(file: &mut File) -> io::Result<Self> {
        let header = Self::load_header(file)?;
        if !header.is_valid() {
            Err(io::Error::other("Wrong archive header"))
        } else {
            file.seek(io::SeekFrom::Start(header.offset))?;
            let directory = Directory::read_from_stream_buffered(file)?;
            Ok(directory)
        }
    }

    fn load_header(file: &mut File) -> io::Result<ArchiveHeader> {
        file.seek(io::SeekFrom::End(-(mem::size_of::<ArchiveHeader>() as i64)))?;
        Ok(ArchiveHeader::read_from_stream_unbuffered(file)?)
    }
}

#[derive(Debug)]
pub struct PackageBuilder {
    file: File,
    directory: Directory,
}

const DATA_ALIGMENT: u64 = 4096;
const VERSION: u32 = 3;
const MAGICK: [u8; 4] = *b"KRPK";

#[derive(Debug, Readable, Writable)]
struct ArchiveHeader {
    pub offset: u64,
    pub version: u32,
    pub magick: [u8; 4],
}

impl ArchiveHeader {
    pub fn new(offset: u64) -> Self {
        Self {
            offset,
            version: VERSION,
            magick: MAGICK,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.magick == MAGICK && self.version == VERSION
    }
}

impl PackageBuilder {
    pub fn new<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        Ok(Self {
            file: File::create(path)?,
            directory: Default::default(),
        })
    }

    pub fn pack(&mut self, reference: AssetReference, data: &[u8]) -> io::Result<()> {
        let offset = self.align_file()?;
        if data.len() <= (DATA_ALIGMENT as usize) {
            self.file.write_all(data)?;
            self.directory.assets.insert(
                reference,
                AssetHeader {
                    offset,
                    size: data.len() as u64,
                    packed: None,
                },
            );
        } else {
            let mut packer = zstd::stream::Encoder::new(Cursor::new(Vec::new()), 19)?;
            packer.write_all(data)?;
            let packed = packer.finish()?.into_inner();
            self.file.write_all(&packed)?;
            self.directory.assets.insert(
                reference,
                AssetHeader {
                    offset,
                    size: data.len() as u64,
                    packed: Some(packed.len() as u64),
                },
            );
        }
        Ok(())
    }

    pub fn finish(&mut self) -> io::Result<()> {
        let offset = self.align_file()?;
        let bytes = self.directory.write_to_vec()?;
        self.file.write_all(&bytes)?;
        let header = ArchiveHeader::new(offset).write_to_vec()?;
        assert_eq!(header.len(), mem::size_of::<ArchiveHeader>());
        self.file.write_all(&header)?;
        Ok(())
    }

    fn align_file(&mut self) -> io::Result<u64> {
        let offset = self.file.stream_position()?;
        let new_size = offset.align(DATA_ALIGMENT);
        self.file.set_len(new_size)?;
        self.file.seek(io::SeekFrom::End(0))
    }
}

pub struct PackedArchive {
    mmap: Mmap,
    directory: Directory,
}

impl PackedArchive {
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let directory = Directory::load(&mut file)?;
        let mmap = unsafe { MmapOptions::new().map(&file) }?;
        Ok(Self { mmap, directory })
    }

    fn read_all<R: Read>(mut r: R, size: usize) -> io::Result<Bytes> {
        let mut data = vec![0u8; size];
        r.read_exact(&mut data)?;
        Ok(data.into())
    }
}

#[async_trait]
impl ArchiveLoad for PackedArchive {
    async fn load(&self, reference: AssetReference) -> io::Result<Bytes> {
        let header = self.directory.assets.get(&reference).ok_or(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Asset {} not found", reference),
        ))?;
        let data = if let Some(packed) = header.packed {
            let data = &self.mmap[header.offset as usize..(header.offset + packed) as usize];
            let data = Cursor::new(unsafe { slice::from_raw_parts(data.as_ptr(), data.len()) });
            Self::read_all(zstd::stream::Decoder::new(data)?, packed as _)?
        } else {
            let data = &self.mmap[header.offset as usize..(header.offset + header.size) as usize];
            data.to_vec().into()
        };
        Ok(data)
    }
}

impl Archive for PackedArchive {
    fn exist(&self, reference: AssetReference) -> bool {
        self.directory.assets.contains_key(&reference)
    }
}
