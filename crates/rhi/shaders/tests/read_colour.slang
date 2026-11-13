// A test shader (design 0002, "Test plan"): draws a triangle that covers
// the whole attachment, whose fragment shader writes the colour at the
// address in its push data, so a test can check that data written on the
// CPU, such as a transient allocation, reached the draw that reads it.

struct PushData
{
    float4* colour;
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
float4 fragmentMain() : SV_Target
{
    return *push.colour;
}
