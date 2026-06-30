use burn_backend::{BoolStore, DType, DeviceOps, DeviceSettings};
use cubecl_common::device::{Device, DeviceId};

// It is not clear if CUDA has a limit on the number of bindings it can hold at
// any given time, but it's highly unlikely that it's more than this. We can
// also assume that we'll never have more than this many bindings in flight,
// so it's 'safe' to store only this many bindings.
pub const CUDA_MAX_BINDINGS: u32 = 1024;

#[derive(new, Clone, PartialEq, Eq, Default, Hash)]
pub struct CudaDevice {
    pub index: usize,
}

impl core::fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cuda({})", self.index)
    }
}

impl Device for CudaDevice {
    fn from_id(device_id: DeviceId) -> Self {
        Self {
            index: device_id.index_id as usize,
        }
    }

    fn to_id(&self) -> DeviceId {
        DeviceId {
            type_id: 0,
            index_id: self.index as u16,
        }
    }
}

/// Burn [`DeviceOps`] impl for `CudaDevice`. Mirrors the cubecl-cuda defaults:
/// f32 floats, i32 ints, u8-packed booleans, no complex dtype.
impl DeviceOps for CudaDevice {
    fn defaults(&self) -> DeviceSettings {
        DeviceSettings::new(
            DType::F32,
            DType::I32,
            DType::Bool(BoolStore::U8),
            None::<DType>,
            Default::default(),
        )
    }
}
