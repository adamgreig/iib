import sys
import colorsys
import scipy.stats
import numpy as np
import pyopencl as cl
from PIL import Image

fname = "light_pattern.png"

# Set up OpenCL
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

# Set up data on the host
light = np.array(Image.open(fname)).astype(np.float32).reshape(512*512) / 255.0
sigs_a = np.zeros(512*512, np.float32)
sigs_b = np.zeros(512*512, np.float32)
output = np.zeros(512*512, np.float32)
image = np.empty(512*512*4, np.uint8)
#sigs_a = np.array(Image.open(fname)).astype(np.float32).reshape(512*512)/255.0
#sigs_a = 1.0 - sigs_a

kernel = scipy.stats.norm(scale=0.8).pdf(range(4)).astype(np.float32)
h2r = colorsys.hsv_to_rgb
colour_lut = [h2r((1.0-(x/255.0)) * 2.0/3.0, 1.0, 1.0) for x in xrange(256)]
colour_lut = np.array(colour_lut, np.float32).reshape(256*3)

# Set up OpenCL buffers for the data
light_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=light)
sigs_in_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=sigs_a)
sigs_out_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=sigs_b)
output_buf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=output)
imagefmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNORM_INT8)
image_buf = cl.Image(ctx, mf.WRITE_ONLY, imagefmt, (512, 512), None)

# Set up strings to hold the constants, to be dumped into the CL code
# (this is much faster than passing them as buffers)
kernel_str =  "__constant float conv_kernel[4] = {"
kernel_str += ",".join(str(k)+"f" for k in kernel) + "};\n"
lut_str = "__constant float colour_lut[768] = {"
lut_str += ",".join(str(k)+"f" for k in colour_lut) + "};\n"

# Read in the CL code and build it
with open("edgedetect.cl") as f:
    progstr = kernel_str + lut_str + f.read()
program = cl.Program(ctx, progstr).build()

# Run the simulation
iterations = 100
for iteration in xrange(iterations):
    program.cell(queue, (512, 512), None, light_buf, sigs_in_buf, sigs_out_buf, output_buf)
    program.diffuse(queue, (512, 512), (256, 1), sigs_out_buf, sigs_in_buf)
    program.diffuse(queue, (512, 512), (1, 256), sigs_in_buf, sigs_out_buf)
    sigs_in_buf, sigs_out_buf = sigs_out_buf, sigs_in_buf

    program.colour(queue, (512, 512), (256, 1), output_buf, image_buf)
    cl.enqueue_copy(queue, image, image_buf, origin=(0,0), region=(512,512)).wait()
    img = Image.fromarray(image.reshape((512, 512, 4)))
    img.save("output/{0:05d}.png".format(iteration))
    if iteration % 100 == 0:
        sys.stdout.write(str(iteration*100/iterations)+"%\r")
        sys.stdout.flush()
