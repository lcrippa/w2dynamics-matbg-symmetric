"""Various supporting functionality and common functionality of the
driver scripts cthyb and DMFT.py"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np


def decomplexify(arr):
    """Takes a complex array-like arr and returns it as a real array with
    an axis for real/imag. part appended.
    """
    return np.concatenate((np.real(arr)[..., np.newaxis],
                           np.imag(arr)[..., np.newaxis]),
                          axis=-1)


def diagonal_covariance(dsample):
    """Takes a complex DistributedSample with data dimensions (iw, band,
    spin, band, spin) and returns the band/spin-diagonal real
    covariance of the mean with dimensions (band, spin, iw1, part1,
    iw2, part2)
    """
    diagsample = (dsample
                  # to (iw, spin, spin, band)
                  .apply(lambda qty:
                         np.diagonal(qty,
                                     axis1=1,
                                     axis2=3))
                  # to (iw, band, spin)
                  .apply(lambda qty:
                         np.diagonal(qty,
                                     axis1=1,
                                     axis2=2))
                  # positive iw only
                  .apply(lambda qty: qty[qty.shape[0]//2:]))
    return np.array([[diagsample
                      .apply(lambda qty, band=b, spin=s: qty[:, band, spin])
                      .apply(decomplexify)
                      .cov_of_mean()
                      for s in range(diagsample.local.shape[-1])]
                     for b in range(diagsample.local.shape[-2])])


def check_stop_iteration(cfg, last_iter_time):
    from os import environ
    from pathlib import Path
    from datetime import datetime, timedelta

    if cfg["General"]["SLURMStopFile"]:
        try:
            open(environ["SLURM_JOB_ID"] + ".stop", "rb")
            return True
        except Exception:
            pass

        try:
            open(environ["SLURM_JOB_ID"] + ".skip", "rb")
            return True
        except Exception:
            pass
        

    if cfg["General"]["SLURMOnlyContinueIfTime"] >= 0:
        # uses end time provided in env e.g. by
        # > export SLURM_JOB_END_DATETIME=$(squeue --local -h -j "${SLURM_JOB_ID}" -o '%e')
        try:
            leftseconds = (datetime.fromisoformat(environ["SLURM_JOB_END_DATETIME"])
                           - datetime.now()) / timedelta(seconds=1)
            if (last_iter_time >= leftseconds - cfg["General"]["SLURMOnlyContinueIfTime"]):
                try:
                    Path(environ["SLURM_JOB_ID"] + ".stop").touch()
                except Exception:
                    pass
                return True
        except Exception:
            pass

    return False
