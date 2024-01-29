#!/usr/bin/env python

"""Tests for `snntorch` package."""

import pytest
import slax as sl
import jax.numpy as jnp


@pytest.fixture(scope="module")
def input_():
    return jnp.array([0.25, 0]).reshape(-1,1)


def test_fire():
    lif = sl.LIF(tau=2.)
    input_large = lif.v_threshold * 10
    assert lif(input_large) == 1.
