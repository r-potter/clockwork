// A test shader (design 0002, "Test plan"): writes each texel's
// coordinates into a storage view, through its slot in the resource heap,
// so a test can read back that a dispatch reached the view.

import heap;

struct PushData
{
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
        destination[id.xy] = float4(id.x, id.y, id.x + id.y, 1.0);
    }
}
