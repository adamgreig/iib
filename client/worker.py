#!/usr/bin/env python
from redis import Redis
from rq import Connection, Queue, Worker
import sys

# Preload libraries
import iib.simulation
import numpy

host = "localhost"
if len(sys.argv) > 1:
    host = sys.argv[1]

with Connection(Redis(host=host, decode_responses=True)):
    Worker([Queue()]).work()
