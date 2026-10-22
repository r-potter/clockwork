// Draws a triangle with a different colour at each corner.
//
// There are no vertex buffers: the vertex shader looks up each corner's
// position and colour by its vertex index, so the triangle is drawn with
// three vertices and no bound resources.

// Clip-space positions. Vulkan's clip space has +Y pointing down, so the
// first corner is at the top.
static const float2 positions[3] = {
    float2(0.0, -0.5),
    float2(0.5, 0.5),
    float2(-0.5, 0.5),
};

// Linear RGB colours, interpolated across the triangle.
static const float3 colours[3] = {
    float3(1.0, 0.0, 0.0),
    float3(0.0, 1.0, 0.0),
    float3(0.0, 0.0, 1.0),
};

struct VertexOutput
{
    float4 position : SV_Position;
    float3 colour : COLOR;
};

[shader("vertex")]
VertexOutput vertexMain(uint vertexIndex: SV_VertexID)
{
    VertexOutput output;
    output.position = float4(positions[vertexIndex], 0.0, 1.0);
    output.colour = colours[vertexIndex];
    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target
{
    return float4(input.colour, 1.0);
}
