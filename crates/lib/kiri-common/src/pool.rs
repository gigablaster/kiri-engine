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

use std::{
    fmt::{Debug, Display},
    hash::Hash,
    marker::PhantomData,
};

const DEFAULT_SPACE: usize = 4096;
const GENERATION_BITS: u32 = 14;
const INDEX_BITS: u32 = 32 - GENERATION_BITS;
const INDEX_MASK: u32 = (1 << INDEX_BITS) - 1;
const GENERATION_MASK: u32 = u32::MAX - INDEX_MASK;
const MAX_INDEX: u32 = (1 << INDEX_BITS) - 1;
const MAX_GENERATION: u32 = 1 << GENERATION_BITS;

#[derive(Debug)]
pub struct Handle<T> {
    data: u32,
    _phantom: PhantomData<T>,
}

unsafe impl<T> Send for Handle<T> {}
unsafe impl<T> Sync for Handle<T> {}

#[allow(clippy::non_canonical_clone_impl)]
impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data,
            _phantom: PhantomData,
        }
    }
}

impl<T> Copy for Handle<T> {}

impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<T> Eq for Handle<T> {}

impl<T> Hash for Handle<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data.hash(state);
    }
}

impl<T> Handle<T> {
    pub fn into_another<U>(&self) -> Handle<U> {
        Handle {
            data: self.data,
            _phantom: PhantomData,
        }
    }

    pub fn new(index: u32, generation: u32) -> Self {
        assert!(index < MAX_INDEX);
        assert!(generation < MAX_GENERATION);
        Self {
            data: (generation << INDEX_BITS) | index,
            _phantom: PhantomData,
        }
    }

    pub fn invalid() -> Self {
        Self {
            data: u32::MAX,
            _phantom: PhantomData,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.data != u32::MAX
    }

    pub fn index(&self) -> u32 {
        self.data & INDEX_MASK
    }

    pub fn generation(&self) -> u32 {
        (self.data & GENERATION_MASK) >> INDEX_BITS
    }
}

impl<T> Default for Handle<T> {
    fn default() -> Self {
        Self::invalid()
    }
}

impl<T> Display for Handle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(idx: {} gen: {})", self.index(), self.generation())
    }
}

impl<T> From<Handle<T>> for u32 {
    fn from(value: Handle<T>) -> Self {
        value.data
    }
}

impl<T> From<u32> for Handle<T> {
    fn from(value: u32) -> Self {
        Handle {
            data: value,
            _phantom: PhantomData,
        }
    }
}

pub trait PoolValueWrapper<T>: Debug {
    type Wrapped: Debug;
    fn wrap(value: T) -> Self::Wrapped;
    fn get(wrapped: &Self::Wrapped) -> Option<&T>;
    fn get_mut(wrapped: &mut Self::Wrapped) -> Option<&mut T>;
    fn unwrap(wrapped: Self::Wrapped) -> Option<T>;
    fn has_value(wrapped: &Self::Wrapped) -> bool;
    fn replace(wrpapped: &mut Self::Wrapped, value: T) -> T;
    fn take(wrapped: &mut Self::Wrapped) -> T;
}

#[derive(Debug)]
pub struct OptionPoolStrategy<T> {
    _marker: PhantomData<T>,
}

impl<T: Debug> PoolValueWrapper<T> for OptionPoolStrategy<T> {
    type Wrapped = Option<T>;

    fn wrap(value: T) -> Self::Wrapped {
        Some(value)
    }

    fn get(wrapped: &Self::Wrapped) -> Option<&T> {
        wrapped.as_ref()
    }

    fn unwrap(wrapped: Self::Wrapped) -> Option<T> {
        wrapped
    }

    fn has_value(wrapped: &Self::Wrapped) -> bool {
        wrapped.is_some()
    }

    fn get_mut(wrapped: &mut Self::Wrapped) -> Option<&mut T> {
        wrapped.as_mut()
    }

    fn replace(wrpapped: &mut Self::Wrapped, value: T) -> T {
        wrpapped.replace(value).unwrap()
    }

    fn take(wrapped: &mut Self::Wrapped) -> T {
        wrapped.take().unwrap()
    }
}

#[derive(Debug)]
pub struct SentinelPoolStrategy<T> {
    _marker: PhantomData<T>,
}

impl<T: Default + Copy + Eq + Debug> PoolValueWrapper<T> for SentinelPoolStrategy<T> {
    type Wrapped = T;

    fn wrap(value: T) -> Self::Wrapped {
        value
    }

    fn get(wrapped: &T) -> Option<&T> {
        Self::has_value(wrapped).then_some(wrapped)
    }

    fn unwrap(wrapped: Self::Wrapped) -> Option<T> {
        Self::has_value(&wrapped).then_some(wrapped)
    }

    fn has_value(wrapped: &Self::Wrapped) -> bool {
        *wrapped != T::default()
    }

    fn get_mut(wrapped: &mut Self::Wrapped) -> Option<&mut T> {
        Self::has_value(wrapped).then_some(wrapped)
    }

    fn replace(wrpapped: &mut Self::Wrapped, value: T) -> T {
        let old = *wrpapped;
        *wrpapped = value;
        old
    }

    fn take(wrapped: &mut Self::Wrapped) -> T {
        let old = *wrapped;
        *wrapped = T::default();
        old
    }
}

#[derive(Debug)]
pub struct Pool<T, Wrapper: PoolValueWrapper<T> = OptionPoolStrategy<T>> {
    data: Vec<Wrapper::Wrapped>,
    generations: Vec<u32>,
    empty: Vec<u32>,
}

impl<T, Wrapper: PoolValueWrapper<T>> Pool<T, Wrapper> {
    pub fn new() -> Self {
        Self {
            data: Vec::with_capacity(DEFAULT_SPACE),
            generations: Vec::with_capacity(DEFAULT_SPACE),
            empty: Vec::with_capacity(DEFAULT_SPACE),
        }
    }

    pub fn push(&mut self, data: T) -> Handle<T> {
        if let Some(slot) = self.empty.pop() {
            self.data[slot as usize] = Wrapper::wrap(data);
            Handle::new(slot, self.generations[slot as usize])
        } else {
            let index = self.generations.len();
            if index == u32::MAX as _ {
                panic!("Too many items in HandleContainer.");
            }
            self.generations.push(0);
            self.data.push(Wrapper::wrap(data));
            Handle::new(index as u32, 0)
        }
    }

    pub fn get(&self, handle: Handle<T>) -> Option<&T> {
        if self.is_handle_valid(&handle) {
            let index = handle.index() as usize;
            Some(Wrapper::get(&self.data[index]).unwrap())
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, handle: Handle<T>) -> Option<&mut T> {
        if self.is_handle_valid(&handle) {
            let index = handle.index() as usize;
            Some(Wrapper::get_mut(&mut self.data[index]).unwrap())
        } else {
            None
        }
    }

    pub fn replace(&mut self, handle: Handle<T>, data: T) -> Option<T> {
        if self.is_handle_valid(&handle) {
            let index = handle.index() as usize;
            Some(Wrapper::replace(&mut self.data[index], data))
        } else {
            None
        }
    }

    pub fn remove(&mut self, handle: Handle<T>) -> Option<T> {
        if self.is_handle_valid(&handle) {
            let index = handle.index() as usize;
            self.generations[index] = self.generations[index].wrapping_add(1) % MAX_GENERATION;
            self.empty.push(index as _);
            return Some(Wrapper::take(&mut self.data[index]));
        }

        None
    }

    pub fn is_handle_valid(&self, handle: &Handle<T>) -> bool {
        let index = handle.index() as usize;
        index < self.generations.len() && self.generations[index] == handle.generation()
    }

    pub fn iter(&self) -> Iter<T, Wrapper> {
        Iter {
            container: self,
            current: 0,
        }
    }

    pub fn enumerate(&self) -> EnumerateHandlesIter<T, Wrapper> {
        EnumerateHandlesIter {
            container: self,
            current: 0,
        }
    }

    pub fn drain(&mut self) -> Drain<T, Wrapper> {
        Drain {
            data: std::mem::take(&mut self.data),
            current: 0,
        }
    }

    pub fn for_each_mut<OP: Fn(&mut T)>(&mut self, op: OP) {
        for index in 0..self.data.len() {
            if let Some(data) = Wrapper::get_mut(&mut self.data[index]) {
                op(data);
            }
        }
    }
}

impl<T, Wrapper: PoolValueWrapper<T>> Default for Pool<T, Wrapper> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Iter<'a, T, Wrapper: PoolValueWrapper<T>> {
    container: &'a Pool<T, Wrapper>,
    current: usize,
}

pub struct EnumerateHandlesIter<'a, T, Wrapper: PoolValueWrapper<T>> {
    container: &'a Pool<T, Wrapper>,
    current: usize,
}

pub struct Drain<T, Wrapper: PoolValueWrapper<T>> {
    data: Vec<Wrapper::Wrapped>,
    current: usize,
}

impl<'a, T, Wrapper: PoolValueWrapper<T>> Iterator for Iter<'a, T, Wrapper> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        while self.current != self.container.data.len()
            && !Wrapper::has_value(&self.container.data[self.current])
        {
            self.current += 1;
        }
        if self.current == self.container.data.len() {
            return None;
        }
        let result = Wrapper::get(&self.container.data[self.current]);
        self.current += 1;

        result
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.container.data.len() - self.container.empty.len() - self.current;

        (size, Some(size))
    }
}

impl<'a, T, Wrapper: PoolValueWrapper<T>> Iterator for EnumerateHandlesIter<'a, T, Wrapper> {
    type Item = (Handle<T>, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        while self.current != self.container.data.len()
            && !Wrapper::has_value(&self.container.data[self.current])
        {
            self.current += 1;
        }
        if self.current == self.container.data.len() {
            return None;
        }
        let result = (
            Handle::new(
                self.current as u32,
                self.container.generations[self.current],
            ),
            Wrapper::get(&self.container.data[self.current]).unwrap(),
        );
        self.current += 1;

        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.container.data.len() - self.container.empty.len() - self.current;

        (size, Some(size))
    }
}

impl<T, Wrapper: PoolValueWrapper<T>> Iterator for Drain<T, Wrapper> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        while self.current != self.data.len() && !Wrapper::has_value(&self.data[self.current]) {
            self.current += 1;
        }
        if self.current == self.data.len() {
            return None;
        }
        let result = Wrapper::take(&mut self.data[self.current]);
        self.current += 1;

        Some(result)
    }
}

#[derive(Debug)]
pub struct HotColdPool<T, U, WrapperT = OptionPoolStrategy<T>, WrapperU = OptionPoolStrategy<U>>
where
    WrapperT: PoolValueWrapper<T>,
    WrapperU: PoolValueWrapper<U>,
{
    hot: Pool<T, WrapperT>,
    cold: Pool<U, WrapperU>,
}

impl<T, U, WrapperT, WrapperU> HotColdPool<T, U, WrapperT, WrapperU>
where
    WrapperT: PoolValueWrapper<T>,
    WrapperU: PoolValueWrapper<U>,
{
    pub fn hot(&self) -> &Pool<T, WrapperT> {
        &self.hot
    }

    pub fn cold(&self) -> &Pool<U, WrapperU> {
        &self.cold
    }

    pub fn push(&mut self, hot: T, cold: U) -> Handle<T> {
        let hot_handle = self.hot.push(hot);
        let cold_handle = self.cold.push(cold);
        #[cfg(debug_assertions)]
        {
            if hot_handle.data != cold_handle.data {
                panic!("Mismatched handles");
            }
        }

        hot_handle
    }

    pub fn get(&self, handle: Handle<T>) -> Option<&T> {
        self.hot.get(handle)
    }

    pub fn get_mut(&mut self, handle: Handle<T>) -> Option<&mut T> {
        self.hot.get_mut(handle)
    }

    pub fn get_cold(&self, handle: Handle<T>) -> Option<&U> {
        self.cold.get(handle.into_another())
    }

    pub fn get_cold_mut(&mut self, handle: Handle<T>) -> Option<&mut U> {
        self.cold.get_mut(handle.into_another())
    }

    pub fn replace(&mut self, handle: Handle<T>, data: T) -> Option<T> {
        self.hot.replace(handle, data)
    }

    pub fn replace_cold(&mut self, handle: Handle<T>, data: U) -> Option<U> {
        self.cold.replace(handle.into_another(), data)
    }

    pub fn remove(&mut self, handle: Handle<T>) -> Option<(T, U)> {
        if let Some(hot) = self.hot.remove(handle) {
            let cold = self
                .cold
                .remove(handle.into_another())
                .expect("Cold data must present under same handle");
            Some((hot, cold))
        } else {
            None
        }
    }

    pub fn iter(&self) -> HotColdIter<T, U, WrapperT, WrapperU> {
        HotColdIter {
            hot: &self.hot,
            cold: &self.cold,
            current: 0,
        }
    }

    pub fn enumerate(&self) -> EnumerateHandleHotColdIter<T, U, WrapperT, WrapperU> {
        EnumerateHandleHotColdIter {
            hot: &self.hot,
            cold: &self.cold,
            current: 0,
        }
    }

    pub fn drain(&mut self) -> HotColdDrain<T, U, WrapperT, WrapperU> {
        HotColdDrain {
            hot: std::mem::take(&mut self.hot.data),
            cold: std::mem::take(&mut self.cold.data),
            current: 0,
        }
    }
}

pub struct HotColdIter<'a, T, U, WrapperT: PoolValueWrapper<T>, WrapperU: PoolValueWrapper<U>> {
    hot: &'a Pool<T, WrapperT>,
    cold: &'a Pool<U, WrapperU>,
    current: usize,
}

pub struct EnumerateHandleHotColdIter<
    'a,
    T,
    U,
    WrapperT: PoolValueWrapper<T>,
    WrapperU: PoolValueWrapper<U>,
> {
    hot: &'a Pool<T, WrapperT>,
    cold: &'a Pool<U, WrapperU>,
    current: usize,
}

impl<'a, T, U, WrapperT, WrapperU> Iterator for HotColdIter<'a, T, U, WrapperT, WrapperU>
where
    WrapperT: PoolValueWrapper<T>,
    WrapperU: PoolValueWrapper<U>,
{
    type Item = (&'a T, &'a U);

    fn next(&mut self) -> Option<Self::Item> {
        while self.current != self.hot.data.len()
            && !WrapperT::has_value(&self.hot.data[self.current])
        {
            self.current += 1;
        }
        if self.current == self.hot.data.len() {
            return None;
        }
        let result = Some((
            WrapperT::get(&self.hot.data[self.current]).unwrap(),
            WrapperU::get(&self.cold.data[self.current]).unwrap(),
        ));
        self.current += 1;

        result
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.hot.data.len() - self.hot.empty.len() - self.current;

        (size, Some(size))
    }
}

impl<'a, T, U, WrapperT, WrapperU> Iterator
    for EnumerateHandleHotColdIter<'a, T, U, WrapperT, WrapperU>
where
    WrapperT: PoolValueWrapper<T>,
    WrapperU: PoolValueWrapper<U>,
{
    type Item = (Handle<T>, &'a T, &'a U);

    fn next(&mut self) -> Option<Self::Item> {
        while self.current != self.hot.data.len()
            && !WrapperT::has_value(&self.hot.data[self.current])
        {
            self.current += 1;
        }
        if self.current == self.hot.data.len() {
            return None;
        }
        let result = Some((
            Handle::new(self.current as u32, self.hot.generations[self.current]),
            WrapperT::get(&self.hot.data[self.current]).unwrap(),
            WrapperU::get(&self.cold.data[self.current]).unwrap(),
        ));
        self.current += 1;

        result
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.hot.data.len() - self.hot.empty.len() - self.current;

        (size, Some(size))
    }
}

impl<T, U, WrapperT, WrapperU> Default for HotColdPool<T, U, WrapperT, WrapperU>
where
    WrapperT: PoolValueWrapper<T>,
    WrapperU: PoolValueWrapper<U>,
{
    fn default() -> Self {
        Self {
            hot: Default::default(),
            cold: Default::default(),
        }
    }
}

pub struct HotColdDrain<T, U, WrapperT: PoolValueWrapper<T>, WrapperU: PoolValueWrapper<U>> {
    hot: Vec<WrapperT::Wrapped>,
    cold: Vec<WrapperU::Wrapped>,
    current: usize,
}

impl<T, U, WrapperT: PoolValueWrapper<T>, WrapperU: PoolValueWrapper<U>> Iterator
    for HotColdDrain<T, U, WrapperT, WrapperU>
{
    type Item = (T, U);

    fn next(&mut self) -> Option<Self::Item> {
        while self.current != self.hot.len() && !WrapperT::has_value(&self.hot[self.current]) {
            self.current += 1;
        }
        if self.current == self.hot.len() {
            return None;
        }
        let result = Some((
            WrapperT::take(&mut self.hot[self.current]),
            WrapperU::take(&mut self.cold[self.current]),
        ));
        self.current += 1;

        result
    }
}

#[cfg(test)]
mod test {
    use crate::{Handle, Pool};

    #[test]
    fn handle() {
        let handle = Handle::<()>::new(100, 10);
        assert_eq!(100, handle.index());
        assert_eq!(10, handle.generation());
    }

    #[test]
    fn handle_container_push_get() {
        let mut container = Pool::<u32>::new();
        let handle1 = container.push(1);
        let handle2 = container.push(2);
        let handle3 = container.push(3);
        assert_eq!(Some(&1), container.get(handle1));
        assert_eq!(Some(&2), container.get(handle2));
        assert_eq!(Some(&3), container.get(handle3));
    }

    #[test]
    fn reuse_slot() {
        let mut container = Pool::<u32>::new();
        let handle = container.push(1);
        container.remove(handle);
        let handle = container.push(2);
        assert_eq!(1, handle.generation());
        assert_eq!(0, handle.index());
        assert_eq!(Some(&2), container.get(handle));
    }

    #[test]
    fn old_handle_returns_none() {
        let mut container = Pool::<u32>::new();
        let handle1 = container.push(1);
        assert_eq!(Some(1), container.remove(handle1));
        let handle2 = container.push(2);
        assert_eq!(None, container.get(handle1));
        assert_eq!(Some(&2), container.get(handle2));
    }

    #[test]
    fn mutate_by_handle() {
        let mut container = Pool::<u32>::new();
        let handle = container.push(1);
        assert_eq!(Some(&1), container.get(handle));
        assert_eq!(Some(1), container.replace(handle, 2));
        assert_eq!(Some(&2), container.get(handle));
        assert_eq!(Some(2), container.replace(handle, 3));
        assert_eq!(Some(&3), container.get(handle));
    }

    #[test]
    fn iterate_empty() {
        let container = Pool::<u32>::new();
        let cont = container.iter().copied().collect::<Vec<_>>();
        assert!(cont.is_empty());
    }

    #[test]
    fn iterate_full() {
        let mut container = Pool::<u32>::new();
        container.push(1);
        container.push(2);
        container.push(3);
        let cont = container.iter().copied().collect::<Vec<_>>();
        assert_eq!([1, 2, 3].to_vec(), cont);
    }

    #[test]
    fn iterate_hole() {
        let mut container = Pool::<u32>::new();
        container.push(1);
        let handle = container.push(2);
        container.push(3);
        container.remove(handle);
        let cont = container.iter().copied().collect::<Vec<_>>();
        assert_eq!([1, 3].to_vec(), cont);
    }

    #[test]
    fn drain() {
        let mut container = Pool::<u32>::new();
        container.push(1);
        container.push(2);
        container.push(3);

        let cont = container.drain().collect::<Vec<_>>();
        assert_eq!([1u32, 2, 3].to_vec(), cont);
        assert!(container.iter().collect::<Vec<_>>().is_empty());
    }
}
