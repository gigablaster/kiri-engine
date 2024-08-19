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
    cmp,
    fmt::{Debug, Display},
    hash::Hash,
    marker::PhantomData, mem::MaybeUninit,
};

pub trait PoolLimits {
    const DEFAULT_SPACE: usize;

    fn generation_bits() -> u32 {
        14
    }

    fn index_bits() -> u32 {
        32 - Self::generation_bits()
    }

    fn index_mask() -> u32 {
        (1 << Self::index_bits()) - 1
    }

    fn generation_mask() -> u32 {
        u32::MAX - Self::index_mask()
    }

    fn max_index() -> u32 {
        (1 << Self::index_bits()) - 1
    }

    fn max_generation() -> u32 {
        1 << Self::generation_bits()

    }
}

#[derive(Debug)]
pub struct DefaultPoolLimits {}

impl PoolLimits for DefaultPoolLimits {
    const DEFAULT_SPACE: usize = 4096;
}

#[derive(Debug)]
pub struct Handle<T, Limits: PoolLimits = DefaultPoolLimits> {
    data: u32,
    _phantom1: PhantomData<T>,
    _phantom2: PhantomData<Limits>,
}

unsafe impl<T, Limits: PoolLimits> Send for Handle<T, Limits> {}
unsafe impl<T, Limits: PoolLimits> Sync for Handle<T, Limits> {}

#[allow(clippy::non_canonical_clone_impl)]
impl<T, Limits: PoolLimits> Clone for Handle<T, Limits> {
    fn clone(&self) -> Self {
        Self {
            data: self.data,
            _phantom1: PhantomData,
            _phantom2: PhantomData,
        }
    }
}

impl<T, Limits: PoolLimits> Copy for Handle<T, Limits> {}

impl<T, Limits: PoolLimits> PartialEq for Handle<T, Limits> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<T, Limits: PoolLimits> Eq for Handle<T, Limits> {}

impl<T, Limits: PoolLimits> Hash for Handle<T, Limits> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data.hash(state);
    }
}

impl<T, Limits: PoolLimits> PartialOrd for Handle<T, Limits> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T, Limits: PoolLimits> Ord for Handle<T, Limits> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if !self.is_valid() && other.is_valid() {
            std::cmp::Ordering::Less
        } else if self.is_valid() && !other.is_valid() {
            std::cmp::Ordering::Greater
        } else {
            self.index().cmp(&other.index())
        }
    }
}

impl<T, Limits: PoolLimits> Handle<T, Limits> {
    pub fn into_another<U>(&self) -> Handle<U> {
        Handle {
            data: self.data,
            _phantom1: PhantomData,
            _phantom2: PhantomData
        }
    }

    pub fn new(index: u32, generation: u32) -> Self {
        assert!(index < Limits::max_index());
        assert!(generation < Limits::max_generation());
        Self {
            data: (generation << Limits::index_bits()) | index,
            _phantom1: PhantomData,
            _phantom2: PhantomData
        }
    }

    pub fn invalid() -> Self {
        Self {
            data: u32::MAX,
            _phantom1: PhantomData,
            _phantom2: PhantomData
        }
    }

    pub fn is_valid(&self) -> bool {
        self.data != u32::MAX
    }

    pub fn index(&self) -> u32 {
        self.data & Limits::index_mask()
    }

    pub fn generation(&self) -> u32 {
        (self.data & Limits::generation_mask()) >> Limits::index_bits()
    }
}

impl<T, Limits: PoolLimits> Default for Handle<T, Limits> {
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
            _phantom1: PhantomData,
            _phantom2: PhantomData
        }
    }
}

pub trait PoolValueWrapper<T> {
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

pub struct MaybeUninitVauleWrapper<T> {
    _phantom: PhantomData<T>
}

impl<T: Copy> PoolValueWrapper<T> for MaybeUninitVauleWrapper<T> {
    type Wrapped = MaybeUninit<T>;

    fn wrap(value: T) -> Self::Wrapped {
        MaybeUninit::new(value)
    }

    fn get(wrapped: &Self::Wrapped) -> Option<&T> {
        Some(unsafe { wrapped.assume_init_ref() })
    }

    fn get_mut(wrapped: &mut Self::Wrapped) -> Option<&mut T> {
        Some(unsafe {wrapped.assume_init_mut()})
    }

    fn unwrap(wrapped: Self::Wrapped) -> Option<T> {
        Some(unsafe {
            wrapped.assume_init()
        })
    }

    fn has_value(wrapped: &Self::Wrapped) -> bool {
        true
    }

    fn replace(wrpapped: &mut Self::Wrapped, value: T) -> T {
        let old = unsafe { wrpapped.assume_init() };
        *wrpapped = MaybeUninit::new(value);
        old
    }

    fn take(wrapped: &mut Self::Wrapped) -> T {
        unsafe { wrapped.assume_init() }
    }
}


#[derive(Debug)]
pub struct Pool<T, Wrapper: PoolValueWrapper<T> = OptionPoolStrategy<T>, Limits: PoolLimits = DefaultPoolLimits> {
    data: Vec<Wrapper::Wrapped>,
    generations: Vec<u32>,
    empty: Vec<u32>,
    _phantom: PhantomData<Limits>
}

impl<T, Wrapper: PoolValueWrapper<T>, Limits: PoolLimits> Pool<T, Wrapper, Limits> {
    pub fn new() -> Self {
        Self {
            data: Vec::with_capacity(Limits::DEFAULT_SPACE),
            generations: Vec::with_capacity(Limits::DEFAULT_SPACE),
            empty: Vec::with_capacity(Limits::DEFAULT_SPACE),
            _phantom: PhantomData
        }
    }

    pub fn push(&mut self, data: T) -> Handle<T, Limits> {
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

    pub fn get(&self, handle: Handle<T, Limits>) -> Option<&T> {
        if self.is_handle_valid(handle) {
            let index = handle.index() as usize;
            Some(Wrapper::get(&self.data[index]).unwrap())
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, handle: Handle<T, Limits>) -> Option<&mut T> {
        if self.is_handle_valid(handle) {
            let index = handle.index() as usize;
            Some(Wrapper::get_mut(&mut self.data[index]).unwrap())
        } else {
            None
        }
    }

    pub fn replace(&mut self, handle: Handle<T, Limits>, data: T) -> Option<T> {
        if self.is_handle_valid(handle) {
            let index = handle.index() as usize;
            Some(Wrapper::replace(&mut self.data[index], data))
        } else {
            None
        }
    }

    pub fn remove(&mut self, handle: Handle<T, Limits>) -> Option<T> {
        if self.is_handle_valid(handle) {
            let index = handle.index() as usize;
            self.generations[index] = self.generations[index].wrapping_add(1) % Limits::max_generation();
            self.empty.push(index as _);
            return Some(Wrapper::take(&mut self.data[index]));
        }

        None
    }

    pub fn is_handle_valid(&self, handle: Handle<T, Limits>) -> bool {
        let index = handle.index() as usize;
        index < self.generations.len() && self.generations[index] == handle.generation()
    }

    pub fn iter(&self) -> Iter<T, Wrapper, Limits> {
        Iter {
            container: self,
            current: 0,
        }
    }

    pub fn enumerate(&self) -> EnumerateHandlesIter<T, Wrapper, Limits> {
        EnumerateHandlesIter {
            container: self,
            current: 0,
            _phantom: PhantomData
        }
    }

    pub fn drain(&mut self) -> Drain<T, Wrapper, Limits> {
        Drain {
            data: std::mem::take(&mut self.data),
            current: 0,
            _phantom: PhantomData
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

pub struct Iter<'a, T, Wrapper: PoolValueWrapper<T>, Limits: PoolLimits> {
    container: &'a Pool<T, Wrapper, Limits>,
    current: usize,
}

pub struct EnumerateHandlesIter<'a, T, Wrapper: PoolValueWrapper<T>, Limits: PoolLimits> {
    container: &'a Pool<T, Wrapper, Limits>,
    current: usize,
    _phantom: PhantomData<Limits>
}

pub struct Drain<T, Wrapper: PoolValueWrapper<T>, Limits: PoolLimits> {
    data: Vec<Wrapper::Wrapped>,
    current: usize,
    _phantom: PhantomData<Limits>
}

impl<'a, T, Wrapper: PoolValueWrapper<T>, Limits: PoolLimits> Iterator for Iter<'a, T, Wrapper, Limits> {
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

impl<'a, T, Wrapper: PoolValueWrapper<T>, Limits: PoolLimits> Iterator for EnumerateHandlesIter<'a, T, Wrapper, Limits> {
    type Item = (Handle<T, Limits>, &'a T);

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

impl<T, Wrapper: PoolValueWrapper<T>, Limits: PoolLimits> Iterator for Drain<T, Wrapper, Limits> {
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

    pub fn replace_hot_cold(&mut self, handle: Handle<T>, hot: T, cold: U) -> Option<(T, U)> {
        if self.is_handle_valud(handle) {
            Some((
                self.hot.replace(handle, hot).unwrap(),
                self.cold.replace(handle.into_another(), cold).unwrap(),
            ))
        } else {
            None
        }
    }

    pub fn is_handle_valud(&self, handle: Handle<T>) -> bool {
        self.hot.is_handle_valid(handle) && self.cold.is_handle_valid(handle.into_another())
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

    #[test]
    fn sort() {
        let mut handles: Vec<Handle<u32>> =
            vec![Handle::new(2, 0), Handle::default(), Handle::new(1, 1)];
        handles.sort();
        assert_eq!(Handle::default(), handles[0]);
        assert_eq!(Handle::new(1, 1), handles[1]);
        assert_eq!(Handle::new(2, 0), handles[2]);
    }
}
