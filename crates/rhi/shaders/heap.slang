// Typed indices into the descriptor heaps (design 0002, "Binding model").
//
// Shaders name a texture view or sampler by its slot in its heap, the
// DescriptorIndex the RHI returned, held in one of these wrappers. Each is
// 4 bytes, as DescriptorIndex is, so it fits push data and structures the
// CPU writes. Shaders use only these, never DescriptorHandle,
// ResourceDescriptorHeap or a bare uint, so changing how shaders reach the
// heaps changes only this file.

module heap;

// A texture view's slot in the resource heap. T is the view's type in the
// shader, such as Texture2D<float4> for a sampled 2D view, or
// RWTexture2D<float4> for a storage view, and must match the view the slot
// was written for.
public struct ResourceIndex<T : IOpaqueDescriptor>
{
    // The slot, from GpuResources::texture_index.
    public uint index;

    // The view.
    public T get()
    {
        return *DescriptorHandle<T>(uint2(index, 0));
    }
}

// A sampler's slot in the sampler heap.
public struct SamplerIndex
{
    // The slot, from GpuResources::sampler_index.
    public uint index;

    // The sampler.
    public SamplerState get()
    {
        return SamplerDescriptorHeap[index];
    }
}
