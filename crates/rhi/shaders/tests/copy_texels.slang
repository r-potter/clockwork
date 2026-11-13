// A test shader (design 0002, "Test plan"): samples a sampled view at each
// texel's centre and writes the result into a storage view of the same
// size, reaching both views and the sampler through their heap slots.

import heap;

struct PushData
{
    ResourceIndex<Texture2D<float4>> source;
    SamplerIndex sampler;
    ResourceIndex<RWTexture2D<float4>> destination;
};

[[vk::push_constant]] ConstantBuffer<PushData> push;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    RWTexture2D<float4> destination = push.destination.get();
    uint width;
    uint height;
    destination.GetDimensions(width, height);
    if (id.x < width && id.y < height)
    {
        float2 uv = (float2(id.xy) + 0.5) / float2(width, height);
        destination[id.xy] = push.source.get().SampleLevel(push.sampler.get(), uv, 0);
    }
}
