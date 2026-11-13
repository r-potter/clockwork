// A test shader (design 0002, "Test plan"): adds one, atomically, to the
// counter at the pointer in its push data for each workgroup it runs, so a
// test can read back how many workgroups a dispatch ran.

struct PushData
{
    uint* counter;
};

[[vk::push_constant]] ConstantBuffer<PushData> push;

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    InterlockedAdd(*push.counter, 1);
}
