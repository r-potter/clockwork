// A test shader (design 0002, "Test plan"): writes each texel of a storage
// view as the average of the 2x2 block of texels under it in a sampled
// view twice its size, as one pass of generating a mip chain does, with a
// box filter. Each view covers one mip, so the source is read at its own
// level 0, with no sampler.

import heap;

struct PushData
{
    ResourceIndex<Texture2D<float4>> source;
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
        Texture2D<float4> source = push.source.get();
        int2 corner = int2(id.xy * 2);
        float4 sum = source.Load(int3(corner, 0))
            + source.Load(int3(corner + int2(1, 0), 0))
            + source.Load(int3(corner + int2(0, 1), 0))
            + source.Load(int3(corner + int2(1, 1), 0));
        destination[id.xy] = sum * 0.25;
    }
}
