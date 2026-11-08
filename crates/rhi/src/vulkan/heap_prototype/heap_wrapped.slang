// Typed 32-bit indices into the heaps, wrapping ResourceDescriptorHeap.

struct ResourceIndex<T : IOpaqueDescriptor>
{
    uint index;

    T get() { return *DescriptorHandle<T>(uint2(index, 0)); }
}

struct SamplerIndex
{
    uint index;

    SamplerState get() { return SamplerDescriptorHeap[index]; }
}

struct Push
{
    float4* output;
    ResourceIndex<Texture2D<float4>> texture;
    SamplerIndex sampler;
};

[[vk::push_constant]] ConstantBuffer<Push> push;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    Texture2D<float4> texture = push.texture.get();
    push.output[0] = texture.Load(int3(0, 0, 0));
    push.output[1] = texture.SampleLevel(push.sampler.get(), float2(0.5, 0.5), 0);
}

