// A test shader (design 0002, "Test plan"): draws a triangle that covers
// the whole attachment, whose fragment shader samples a sampled view at
// each fragment's lower-right corner, reaching the view and the sampler
// through their heap slots. With a view the attachment's size, that's
// where four texels meet, so what's rendered depends on the sampler: a
// linear one blends the four, and a nearest one picks one.

import heap;

struct PushData
{
    ResourceIndex<Texture2D<float4>> source;
    SamplerIndex sampler;
};

[[vk::push_constant]] ConstantBuffer<PushData> push;

// Clip-space corners of a triangle that covers the clip-space square, -1
// to 1 on each axis, with its other half off-screen.
static const float2 corners[3] = {
    float2(-1.0, -1.0),
    float2(3.0, -1.0),
    float2(-1.0, 3.0),
};

[shader("vertex")]
float4 vertexMain(uint vertexIndex: SV_VertexID) : SV_Position
{
    return float4(corners[vertexIndex], 0.0, 1.0);
}

[shader("fragment")]
float4 fragmentMain(float4 position: SV_Position) : SV_Target
{
    Texture2D<float4> source = push.source.get();
    uint width;
    uint height;
    source.GetDimensions(width, height);
    // A fragment's position is its centre, in pixels, so this samples the
    // corner the texel with the same coordinates shares with the three
    // after it in x and y.
    float2 uv = (position.xy + 0.5) / float2(width, height);
    return source.SampleLevel(push.sampler.get(), uv, 0);
}
