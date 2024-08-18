// Copyright (C) 2024 gigablaster

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

use std::collections::HashMap;

use kiri_backend::RenderContext;

use crate::{Error, RenderTargetManager, TemporaryRenderTarget, TemporaryRenderTargetDesc};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FramePassResourceType {
    RenderTarget(TemporaryRenderTargetDesc),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FramePassResourceDesc<'a> {
    pub name: &'a str,
    pub ty: FramePassResourceType,
}

#[derive(Debug)]
pub enum ResolvedFrameResource<'a> {
    RenderTarget(TemporaryRenderTarget<'a>),
}

/// Проход рендера
///
/// Система не имеет какого либо рендерграфа, потому правильный порядок
/// отправки проходов лежит на клиенте.
pub trait FrameRenderPass {
    /// Список входящих ресурсов необходимых для прохода
    fn required(&self) -> &[FramePassResourceDesc];
    /// Исполняет проход, возвращает список сгенерированных ресурсов
    fn execute(
        &self,
        context: &RenderContext,
        targets: &RenderTargetManager,
        resources: HashMap<&str, &ResolvedFrameResource>,
    ) -> Result<Vec<(FramePassResourceDesc, ResolvedFrameResource)>, Error>;
}

/// Испольняет кадр
///
/// Проходится по всем проходам и выполняет их. Управляет ресурсами во время
/// рендеринга. Ресурс созданный на раннем этапе может быть отдан проходу на
/// позднем этапе.
pub fn execute_frame(
    context: &RenderContext,
    targets: &RenderTargetManager,
    passes: &[Box<dyn FrameRenderPass>],
) -> Result<(), Error> {
    let mut resources = HashMap::new();
    for pass in passes {
        let request = pass.required();
        let mut resolved = HashMap::default();
        for it in request {
            let resource = resources.get(it).unwrap();
            resolved.insert(it.name, resource);
        }
        let generated = pass.execute(context, targets, resolved).unwrap();
        for (name, resource) in generated {
            resources.insert(name, resource);
        }
    }
    Ok(())
}
