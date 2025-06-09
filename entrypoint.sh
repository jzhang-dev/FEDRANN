#!/usr/bin/env bash

export NUMBA_DISABLE_CACHE=1
/usr/bin/time -v SeqNeighbor "$@"