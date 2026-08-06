"""Asynchronous job execution (ADR-0009).

One subprocess per job, drawn from a single FIFO queue, communicating by JSON-lines
events on stdout. Nothing here knows what a job *does*: kinds are looked up in the
handler registry, so adding a kind of work costs one entry and one function.
"""
