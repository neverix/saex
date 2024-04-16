def test_tpu_stuff_works():
    import numpy as np
    import jax
    import jax.numpy as jnp
    
    a = np.array([1, 2, 3]) + 1
    b = jnp.array([-4, -5, -6])
    b = jax.device_put(b, jax.devices("tpu")[0])
    c = np.asarray(a + b).tolist()
    assert c == [-2, -2, -2]
