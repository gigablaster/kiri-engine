// Copyright (C) 2023 gigablaster

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

use std::fmt::Display;

#[derive(Debug, Clone, Copy, Default)]
pub struct GameTime {
    pub delta_time: f32,
    pub raw_delta_time: f32,
    pub frame_number: u32,
    pub total_time: f32,
}

impl Display for GameTime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "(dt: {} raw: {} frame: {} total: {})",
            self.delta_time, self.raw_delta_time, self.frame_number, self.total_time
        )
    }
}

pub const TARGET_FPS: u64 = 60;
const SAMPLES: usize = 15;
const IGNORE: usize = 2;
const COUNT: usize = SAMPLES - IGNORE * 2;
const DEFAULT_DT: f64 = 1.0 / TARGET_FPS as f64;

#[derive(Debug, Clone)]
pub struct TimeFilter {
    raw: [f64; SAMPLES],
    cursor: usize,
    count: u32,
    total: f64,
}

impl TimeFilter {
    pub fn new() -> TimeFilter {
        let raw = [DEFAULT_DT; SAMPLES];

        TimeFilter {
            raw,
            cursor: 0,
            count: 0,
            total: 0.0,
        }
    }

    pub fn sample(&mut self, dt: f64) -> GameTime {
        self.raw[self.cursor] = dt;
        self.cursor += 1;
        self.cursor %= SAMPLES;
        self.total += dt;
        let mut sorted = self.raw;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut average = 0.0;
        (0..COUNT).for_each(|index| {
            average += sorted[index];
        });
        let average = average / COUNT as f64;

        GameTime {
            delta_time: average as f32,
            raw_delta_time: dt as f32,
            frame_number: self.count,
            total_time: self.total as f32,
        }
    }
}

impl Default for TimeFilter {
    fn default() -> Self {
        Self::new()
    }
}
