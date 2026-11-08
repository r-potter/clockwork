// A test shader (design 0002, "Test plan"): writes 0xC10CC0DE through the
// pointer in its push data, so a test can read back that a dispatch ran
// and reached its buffer by address.

struct PushData
{
    uint* output;
};

[[vk::push_constant]] ConstantBuffer<PushData> push;

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    *push.output = 0xC10CC0DE;
}
