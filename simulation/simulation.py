import colour
import genome
import diffusion

import numpy as np
import pyopencl as cl

from PIL import Image

def main():
    wg_size = 128
    g_size  = 512

    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    
    #sigs_a = np.random.random(g_size*g_size*16).astype(np.float32) * 0.8
    sigs_a = np.zeros((g_size, g_size, 16), np.float32)
    sigs_a[206:306,206:306,0] = 1.0
    sigs_a[0:512  ,0:206  ,1] = 1.0
    sigs_a[0:512  ,307:512,1] = 1.0
    sigs_a[0:206  ,206:307,1] = 1.0
    sigs_a[307:512,206:307,1] = 1.0
    sigs_a = sigs_a.reshape(g_size * g_size * 16)
    sigs_b = np.empty(g_size*g_size*16, np.float32)
    
    buf_a = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=sigs_a)
    buf_b = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=sigs_b)

    image = np.empty(512*512*4, np.uint8)
    ifmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNORM_INT8)
    ibuf = cl.Image(ctx, mf.WRITE_ONLY, ifmt, (g_size, g_size), None)

    def dump_image(buf, sig, iteration):
        bsig = str(chr(sig)).encode()
        program.colour(queue, (g_size, g_size), None, buf, bsig, ibuf)
        cl.enqueue_copy(
            queue, image, ibuf, origin=(0,0), region=(g_size, g_size)).wait()
        img = Image.fromarray(image.reshape((g_size, g_size, 4)))
        img.save("output/{0}_{1:05d}.png".format(sig, iteration))

    test_genome = "+0106-1609-0313"
    test_sigmas = [1.0, 5.0] * 8
    progstr = genome.genome_cl(test_genome)
    progstr += diffusion.diffusion_cl(test_sigmas, wg_size)
    progstr += colour.colour_cl()
    program = cl.Program(ctx, progstr).build()

    dump_image(buf_a, 0, 0)
    dump_image(buf_a, 1, 0)

    iterations = 500
    for iteration in range(iterations):
        program.genome(queue, (g_size, g_size), None, buf_a, buf_b).wait()
        program.diffusion(queue, (g_size, g_size), (wg_size, 1), buf_b, buf_a)
        program.diffusion(queue, (g_size, g_size), (1, wg_size), buf_a, buf_b).wait()
        buf_a, buf_b = buf_b, buf_a
        dump_image(buf_a, 0, iteration+1) 
        dump_image(buf_a, 1, iteration+1) 


if __name__ == "__main__":
    main()
