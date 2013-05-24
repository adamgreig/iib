import time
import sqlite3
import datetime
from rq import Queue, use_connection
from rq.job import Job
from redis import Redis

from iib.server import evolution


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


def queue_first_gen(conns):
    jobs = []
    cohort_size = int(input("Enter initial cohort size: "))
    cohort = evolution.first_generation(cohort_size)
    print("Enqueing first generation jobs...")
    for genome in cohort:
        job = conns.q.enqueue_call(func='iib.client.process.rq_job',
                                   kwargs={"generation": 1, "genome": genome},
                                   result_ttl=-1)
        jobs.append(job)
        save_task(conns, 1, job._id, genome)
    conns.sql_con.commit()
    return jobs


def fetch_jobs_from_gen(conns, gen):
    jobs = []
    job_ids = get_jobs(conns, gen)
    for job_id in job_ids:
        job = Job.fetch(job_id)
        jobs.append(job)
    return jobs


def wait_for_gen(conns, gen):
    print("Waiting for generation {0}...".format(gen))
    jobs = fetch_jobs_from_gen(conns, gen)
    saved_jobs = []
    t0 = time.time()
    while True:
        time.sleep(5)
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

        ts = datetime.datetime.now().isoformat()
        print("[{0}] generation {1}: {2} finished, {3} started, "
              "{4} queued, {5} failed jobs"
              .format(ts, gen, finished, started, queued, failed))

        if finished != len(jobs):
            td = time.time() - t0
            if td > 600:
                print("It's been ten minutes! Moving on with what we've got.")
                return
        else:
            print("All jobs finished, proceeding...")
            return


def main():
    redis_conn = Redis(decode_responses=True)
    use_connection(redis_conn)
    q = Queue()
    sql_con, sql_cur = setup_db()
    conns = Connections(q, sql_con, sql_cur)

    gen = int(input("Enter current generation (0 to start from scratch): "))

    if gen == 0:
        queue_first_gen(conns)
        gen = 1

    while True:
        wait_for_gen(conns, gen)


if __name__ == "__main__":
    main()
