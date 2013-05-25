import time
import logging
import sqlite3
from rq import Queue, use_connection
from rq.job import Job
from redis import Redis

from iib.server import evolution
from iib.server.notify import send_message


logger = logging.getLogger('iib')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
fh = logging.FileHandler("iib_server.log")
formatter = logging.Formatter("%(asctime)s: %(levelname)s: %(message)s")
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)


def setup_db():
    connection = sqlite3.connect("iib.sqlite")
    cur = connection.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS iib "
        "(id INTEGER PRIMARY KEY, generation INTEGER, job TEXT(36),"
        " genome TEXT(250), score REAL)")
    return connection, cur


def save_task(conns, generation, job, genome):
    conns.sql_cur.execute("INSERT INTO iib (generation, job, genome) VALUES "
                          "(?, ?, ?)", (generation, job, genome))


def save_score(conns, job, score):
    cur = conns.sql_cur
    cur.execute("UPDATE iib SET score = (?) WHERE job = (?)", (score, job))
    conns.sql_con.commit()


def get_jobs(conns, gen):
    conns.sql_cur.execute("SELECT job FROM iib WHERE generation = (?)", (gen,))
    rows = [r[0] for r in conns.sql_cur.fetchall()]
    return rows


class Connections:
    def __init__(self, q, sql_con, sql_cur):
        self.q = q
        self.sql_con = sql_con
        self.sql_cur = sql_cur


def queue_cohort(conns, gen, genomes):
    jobs = []
    for genome in genomes:
        job = conns.q.enqueue_call(func='iib.client.process.rq_job',
                                   kwargs={"generation": gen,
                                           "genome": genome},
                                   result_ttl=-1)
        jobs.append(job)
        save_task(conns, gen, job._id, genome)
    conns.sql_con.commit()
    return jobs


def fetch_jobs_from_gen(conns, gen):
    jobs = []
    job_ids = get_jobs(conns, gen)
    for job_id in job_ids:
        job = Job.fetch(job_id)
        jobs.append(job)
    return jobs


def fetch_results_from_gen(conns, gen):
    conns.sql_cur.execute("SELECT genome, score FROM iib WHERE generation = ?"
                          " AND score", (gen,))
    return conns.sql_cur.fetchall()


def wait_for_gen(conns, gen):
    logger.info("Waiting for generation %d...", gen)
    jobs = fetch_jobs_from_gen(conns, gen)
    saved_jobs = []
    t0 = time.time()
    old_finished = 0
    while True:
        finished = 0
        failed = 0
        started = 0
        queued = 0
        for job in jobs:
            if job.is_finished:
                if job not in saved_jobs:
                    save_score(conns, job._id, job.result)
                    saved_jobs.append(job)
                finished += 1
            elif job.is_failed:
                failed += 1
            elif job.is_started:
                started += 1
            elif job.is_queued:
                queued += 1

        logger.info("generation %d: %d finished, %d started, "
                    "%d queued, %d failed jobs",
                    gen, finished, started, queued, failed)

        if finished == len(jobs):
            logger.info("All jobs finished, proceeding...")
            return
        elif finished + failed == len(jobs):
            logger.warn("All jobs finished or failed, proceeding...")
            send_message("Failures", "Generation {0}".format(gen))
            return
        else:
            if old_finished != finished:
                t0 = time.time()
            elif time.time() - t0 > 60:
                logger.warn("It's been a minute with no progress! "
                            "Moving on with what we've got.")
                send_message("Timeout", "Generation {0}".format(gen))
                return

        old_finished = finished
        time.sleep(5)


def main():
    redis_conn = Redis(decode_responses=True)
    use_connection(redis_conn)
    q = Queue()
    sql_con, sql_cur = setup_db()
    conns = Connections(q, sql_con, sql_cur)

    cohort_size = int(input("Enter cohort size: "))
    gen = int(input("Enter current generation (0 to start from scratch): "))

    if gen == 0:
        gen = 1
        cohort = evolution.first_generation(cohort_size)
        queue_cohort(conns, gen, cohort)

    while True:
        wait_for_gen(conns, gen)
        old_cohort = fetch_results_from_gen(conns, gen)
        highscore = max(i[1] for i in old_cohort)
        logger.info("Generation complete. Max score: %.2f", highscore)
        if gen % 10 == 0:
            send_message(
                "Generation Complete",
                "Gen {0}, max score {1:.2f}".format(gen, highscore))
        new_cohort = evolution.new_generation(old_cohort)
        members = len(new_cohort)
        if members < cohort_size:
            # timeout or fails
            new_cohort += evolution.first_generation(cohort_size - members)
        gen += 1
        logger.info("Enqueueing tasks for generation %d...", gen)
        queue_cohort(conns, gen, new_cohort)
        logger.info("Tasks queued, waiting for completion...")


if __name__ == "__main__":
    main()
