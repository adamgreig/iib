#!/usr/bin/env python
from redis import Redis
from rq import Connection, Queue, Worker

# Preload libraries
import iib

with Connection(Redis(decode_responses=True)):
    Worker([Queue()]).work()
