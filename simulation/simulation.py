import colour
import genome
import diffusion

import numpy as np
import pyopencl as cl

from PIL import Image

def main():
    wg_size = 256
    g_size  = 512

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    
    sigs_a = np.ones(g_size*g_size*16, np.float32) * 0.5
    sigs_b = np.empty(g_size*g_size*16, np.float32)
    
    buf_a = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=sigs_a)
    buf_b = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=sigs_b)

    test_genome = "+0605+0111-1505"
    test_sigmas = [1.0, 2.0] * 8
    progstr = genome.genome_cl(test_genome)
    progstr += diffusion.diffusion_cl(test_sigmas, wg_size)
    progstr += colour.colour_cl()
    program = cl.Program(ctx, progstr).build()

    iterations = 100
    for iteration in range(iterations):
        #program.genome(queue, (g_size, g_size), None, buf_a, buf_b)
        #program.diffusion(queue, (g_size, g_size), (wg_size, 1), buf_b, buf_a)
        program.diffusion(queue, (g_size, g_size), (1, wg_size), buf_a, buf_b).wait()
        #buf_a, buf_b = buf_b, buf_a

    #image = np.empty(512*512*4, np.uint8)
    #ifmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNORM_INT8)
    #ibuf = cl.Image(ctx, mf.WRITE_ONLY, ifmt, (g_size, g_size), None)
    
    #program.colour(queue, (g_size, g_size), None, buf_a, b"\x00", ibuf)
    #cl.enqueue_copy(queue, image, ibuf, origin=(0,0), region=(g_size, g_size)).wait()

    #img = Image.fromarray(image.reshape((g_size, g_size, 4)))
    #img.save("simulation.png")

    queue.flush()
    queue.finish()


if __name__ == "__main__":
    main()
