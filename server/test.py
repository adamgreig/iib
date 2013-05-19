from rq import Queue
from redis import Redis

from iib.client import process
from copy import deepcopy


def main():
    redis_conn = Redis()
    q = Queue(connection=redis_conn)
    genome_base = "+0{0}0{1}-8{2}0{3}-0{4}8{5}"
    cfg = deepcopy(process.standard_config)
    cfg["signals"][0]["initial"] = "box"
    cfg["signals"][8]["initial"] = "box"
    cfg["signals"][8]["initial_scale"] = -1.0
    cfg["signals"][8]["initial_offset"] = 1.0
    cfg["dump_final_image"] = True
    for w in range(10):
        g = genome_base.format(2, 7, w, 9, 3, 3)
        cfg["genome"] = g
        q.enqueue_call(func='iib.client.process.rq_job',
                       kwargs={"generation": 0, "config": cfg}, result_ttl=-1)

if __name__ == "__main__":
    main()
