#! /usr/bin/env pytest

import rex
import numpy as np
from openmmtools.states import ThermodynamicState, SamplerState
from openmmtools.mcmc import MCMCMove


def test_replica_workflow():
    flow = rex.ReplicaExchangeWorkflow("rex.yml")


def test_thermodynamic_states():
    flow = rex.ReplicaExchangeWorkflow("rex.yml")
    assert all([isinstance(tstate, ThermodynamicState) for tstate in flow.thermodynamic_states])


def test_sampler_states():
    flow = rex.ReplicaExchangeWorkflow("rex.yml")
    assert all([isinstance(sstate, SamplerState) for sstate in flow.sampler_states])


def test_mcmc_move():
    flow = rex.ReplicaExchangeWorkflow("rex.yml")
    assert isinstance(flow.mcmc_move, MCMCMove)


def test_run():
    flow = rex.ReplicaExchangeWorkflow("rex.yml")
    sampler, _ = flow.multistate_sampler()
    sampler.run(1)
    sampler.run(1)


def test_traj():
    flow = rex.ReplicaExchangeWorkflow("rex.yml")
    flow.to_mdtraj(iterations=np.arange(2))

