// Draws a triangle with a different colour at each corner, reading its
// vertices through the pointer in its push data (design 0002, "The
// triangle"), so drawing it exercises device addresses and the uploader.
//
// `src/triangle.rs` lays out `Vertex` and `PushData` to match.

// One corner of the triangle. Both fields are `float4`s, so the layout is
// the same in every buffer layout rule, with no padding.
struct Vertex
{
    // Clip-space position. Vulkan's clip space has +Y pointing down.
    float4 position;
    // Linear RGB colour, with alpha, interpolated across the triangle.
    float4 colour;
};

struct PushData
{
    // The three corners.
    Vertex* vertices;
};

[[vk::push_constant]] ConstantBuffer<PushData> push;

struct VertexOutput
{
    float4 position : SV_Position;
    float4 colour : COLOR;
};

[shader("vertex")]
VertexOutput vertexMain(uint vertexIndex: SV_VertexID)
{
    Vertex vertex = push.vertices[vertexIndex];
    VertexOutput output;
    output.position = vertex.position;
    output.colour = vertex.colour;
    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target
{
    return input.colour;
}
