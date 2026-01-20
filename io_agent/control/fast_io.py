"""Fast IO Controller.

Augmenter: handles history and feature augmentation
IO Model: linear Q-function with identity theta_uu

All code is pure JAX (JIT-able and vmap-able).
"""

from typing import Tuple
import chex
import jax
import jax.numpy as jnp
import jax.random as jrd
import optax
from jaxtyping import Array, Float, ArrayLike


# =============================================================================
# Augmenter
# =============================================================================


@chex.dataclass
class AugmenterState:
    """State for step-by-step augmentation."""

    history_obs: Float[Array, "n_past obs_dim"]
    history_act: Float[Array, "n_past act_dim"]


@chex.dataclass
class AugNormState:
    """Running normalization statistics for augmented features (like BatchNorm)."""

    mean: Float[Array, "aug_dim"]
    std: Float[Array, "aug_dim"]

    @staticmethod
    def create(aug_dim: int) -> "AugNormState":
        """Initialize with zeros mean and ones std (no-op normalization)."""
        return AugNormState(
            mean=jnp.zeros((aug_dim,)),
            std=jnp.ones((aug_dim,)),
        )


def update_norm_state(
    norm_state: AugNormState,
    aug_batch: Float[Array, "... aug_dim"],
    momentum: float = 0.0,
) -> AugNormState:
    """Update running mean/std with batch statistics (EMA update).

    Args:
        norm_state: Current running statistics
        aug_batch: Batch of augmented features (any leading dims, last dim is aug_dim)
        momentum: 0 = no normalization (keep mean=0, std=1), 1 = full batch norm

    Returns:
        Updated norm state
    """
    flat_batch = aug_batch.reshape(-1, aug_batch.shape[-1])
    batch_mean = jnp.mean(flat_batch, axis=0)
    batch_std = jnp.std(flat_batch, axis=0) + 1e-8

    new_mean = (1 - momentum) * norm_state.mean + momentum * batch_mean
    new_std = (1 - momentum) * norm_state.std + momentum * batch_std

    return AugNormState(mean=new_mean, std=new_std)


def normalize_aug(
    aug: Float[Array, "... aug_dim"],
    norm_state: AugNormState,
) -> Float[Array, "... aug_dim"]:
    """Normalize augmented features using running statistics."""
    return (aug - norm_state.mean) / norm_state.std


@chex.dataclass(frozen=True)
class Augmenter:
    """Augmenter matching FeatureHandler exactly.

    Feature order (matching FeatureHandler.augment_state):
    1. state * scale_factor (normalized)
    2. state_history (normalized) * scale_factor (if use_state_regressor)
    3. action_history (if use_action_regressor)
    4. bias (if add_bias)
    5. co_product: (state @ state.T).ravel() * scale_factor^2 (if use_co_product)
    6. sinusoidal: cos(1,2,4,8) then sin(1,2,4,8) of (pi * state * scale_factor)
    7. rff: sqrt(2/D) * cos(state * scale_factor @ W + b) (if use_rff)
    8. rbf: exp(-||state - c_i||^2 / (2 * sigma^2)) for random centers (if use_rbf)

    All fields are immutable JAX arrays or Python scalars for JIT compatibility.
    """

    # RFF parameters
    rff_W: Float[Array, "obs_dim rff_dim"]
    rff_b: Float[Array, "rff_dim"]

    # RBF parameters
    rbf_centers: Float[Array, "rbf_n_features obs_dim"]
    rbf_gamma: float  # 1 / (2 * sigma^2)

    # Normalization bounds
    state_low: Float[Array, "obs_dim"]
    state_high: Float[Array, "obs_dim"]

    # Dimensions and config (static for JIT)
    obs_dim: int
    act_dim: int
    n_past: int
    scale_factor: float
    add_bias: bool
    use_state_regressor: bool
    use_action_regressor: bool
    use_co_product: bool
    use_sinusoidal: bool
    sin_freqs: Float[Array, "n_freqs"]
    use_rff: bool
    rff_n_features: int
    use_rbf: bool
    rbf_n_features: int

    # Normalization momentum (fixed constant, 0 = no update)
    norm_momentum: float

    @staticmethod
    def create(
        obs_dim: int,
        act_dim: int,
        state_low: ArrayLike,
        state_high: ArrayLike,
        n_past: int = 4,
        scale_factor: float = 1.0,
        add_bias: bool = True,
        use_state_regressor: bool = True,
        use_action_regressor: bool = True,
        use_co_product: bool = True,
        use_sinusoidal: bool = True,
        sin_freqs: ArrayLike = (1.0, 2.0, 4.0, 8.0),
        use_rff: bool = True,
        rff_n_features: int = 1024,
        rff_sigma: float = 0.1,
        rff_seed: int = 42,
        use_rbf: bool = False,
        rbf_n_features: int = 1024,
        rbf_sigma: float = 0.1,
        norm_momentum: float = 0.0,
    ) -> "Augmenter":
        """Create Augmenter.

        RFF uses JAX random for initialization.
        RBF centers are sampled uniformly from [state_low, state_high].
        """
        key = jrd.PRNGKey(rff_seed)
        k1, k2, k3 = jrd.split(key, 3)

        # RFF initialization
        rff_W = jrd.normal(k1, (obs_dim, rff_n_features)) / rff_sigma
        rff_b = jrd.uniform(k2, (rff_n_features,), minval=0, maxval=2 * jnp.pi)

        # RBF centers: sample uniformly from state space
        state_low_arr = jnp.asarray(state_low)
        state_high_arr = jnp.asarray(state_high)
        rbf_centers = jrd.uniform(
            k3, (rbf_n_features, obs_dim), minval=state_low_arr, maxval=state_high_arr
        )
        rbf_gamma = 1.0 / (2.0 * rbf_sigma**2)

        # Compute aug_dim for normalization initialization
        aug_dim = obs_dim
        if use_state_regressor:
            aug_dim += n_past * obs_dim
        if use_action_regressor:
            aug_dim += n_past * act_dim
        if add_bias:
            aug_dim += 1
        if use_co_product:
            aug_dim += obs_dim * obs_dim
        if use_sinusoidal:
            aug_dim += 2 * len(sin_freqs) * obs_dim
        if use_rff:
            aug_dim += rff_n_features
        if use_rbf:
            aug_dim += rbf_n_features

        return Augmenter(
            rff_W=rff_W,
            rff_b=rff_b,
            rbf_centers=rbf_centers,
            rbf_gamma=float(rbf_gamma),
            state_low=state_low_arr,
            state_high=state_high_arr,
            obs_dim=obs_dim,
            act_dim=act_dim,
            n_past=n_past,
            scale_factor=float(scale_factor),
            add_bias=bool(add_bias),
            use_state_regressor=bool(use_state_regressor),
            use_action_regressor=bool(use_action_regressor),
            use_co_product=bool(use_co_product),
            use_sinusoidal=bool(use_sinusoidal),
            sin_freqs=jnp.asarray(sin_freqs),
            use_rff=bool(use_rff),
            rff_n_features=int(rff_n_features),
            use_rbf=bool(use_rbf),
            rbf_n_features=int(rbf_n_features),
            norm_momentum=float(norm_momentum),
        )

    def reset(self) -> AugmenterState:
        """Create initial augmenter state.

        Matches FeatureHandler.reset_history():
        - state_history: initialized to midpoint of [state_low, state_high]
        - action_history: initialized to zeros
        """
        state_mid = (self.state_high + self.state_low) / 2
        return AugmenterState(
            history_obs=jnp.tile(state_mid, (self.n_past, 1)),
            history_act=jnp.zeros((self.n_past, self.act_dim)),
        )

    def aug_dim(self) -> int:
        """Compute augmented state dimension (matches FeatureHandler.aug_state_size)."""
        dim = self.obs_dim
        if self.use_state_regressor:
            dim += self.n_past * self.obs_dim
        if self.use_action_regressor:
            dim += self.n_past * self.act_dim
        if self.add_bias:
            dim += 1
        if self.use_co_product:
            dim += self.obs_dim * self.obs_dim
        if self.use_sinusoidal:
            dim += 2 * len(self.sin_freqs) * self.obs_dim
        if self.use_rff:
            dim += self.rff_n_features
        if self.use_rbf:
            dim += self.rbf_n_features
        return dim

    def _normalize(
        self, obs: Float[Array, "... obs_dim"]
    ) -> Float[Array, "... obs_dim"]:
        """Normalize observation to [-1, 1] (matches FeatureHandler.normalize)."""
        state_range = self.state_high - self.state_low
        return (obs - self.state_low) / state_range * 2 - 1

    def _rff_transform(
        self, obs_scaled: Float[Array, "obs_dim"]
    ) -> Float[Array, "rff_dim"]:
        """Random Fourier Features (matches FeatureHandler.rff_transform)."""
        return jnp.sqrt(2.0 / self.rff_n_features) * jnp.cos(
            obs_scaled @ self.rff_W + self.rff_b
        )

    def _rbf_transform(
        self, obs: Float[Array, "obs_dim"]
    ) -> Float[Array, "rbf_n_features"]:
        """Radial Basis Function features: exp(-gamma * ||obs - center||^2)."""
        # obs: (obs_dim,), centers: (rbf_n_features, obs_dim)
        diff = obs - self.rbf_centers  # (rbf_n_features, obs_dim)
        sq_dist = jnp.sum(diff**2, axis=1)  # (rbf_n_features,)
        return jnp.exp(-self.rbf_gamma * sq_dist)

    def _augment_raw(
        self,
        obs: Float[Array, "obs_dim"],
        history_obs: Float[Array, "n_past obs_dim"],
        history_act: Float[Array, "n_past act_dim"],
    ) -> Float[Array, "aug_dim"]:
        """Augment observation (raw, before normalization)."""
        obs_norm = self._normalize(obs)
        obs_scaled = obs_norm * self.scale_factor
        scale_sq = self.scale_factor**2
        empty = jnp.zeros((0,))

        # Vectorized sinusoidal
        angles = jnp.pi * obs_scaled * self.sin_freqs[:, None]
        cos_sin = jnp.concatenate((jnp.cos(angles).ravel(), jnp.sin(angles).ravel()))

        return jnp.concatenate(
            (
                obs_scaled,
                self._normalize(history_obs).ravel() * self.scale_factor
                if self.use_state_regressor
                else empty,
                history_act.ravel() if self.use_action_regressor else empty,
                jnp.ones((1,)) if self.add_bias else empty,
                (obs_norm[:, None] * obs_norm[None, :]).ravel() * scale_sq
                if self.use_co_product
                else empty,
                cos_sin if self.use_sinusoidal else empty,
                self._rff_transform(obs_scaled) if self.use_rff else empty,
                self._rbf_transform(obs_norm) if self.use_rbf else empty,
            )
        )

    def _augment(
        self,
        obs: Float[Array, "obs_dim"],
        history_obs: Float[Array, "n_past obs_dim"],
        history_act: Float[Array, "n_past act_dim"],
        norm_state: AugNormState,
    ) -> Float[Array, "aug_dim"]:
        """Augment observation with normalization."""
        aug = self._augment_raw(obs, history_obs, history_act)
        return (aug - norm_state.mean) / norm_state.std

    def step(
        self,
        state: AugmenterState,
        obs: Float[Array, "obs_dim"],
        act: Float[Array, "act_dim"],
        done: Float[Array, ""],
        norm_state: AugNormState,
    ) -> Tuple[Float[Array, "aug_dim"], AugmenterState]:
        """Single step augmentation with history update."""
        aug_obs = self._augment(obs, state.history_obs, state.history_act, norm_state)

        # Compute updated history (normal case)
        updated_obs = jnp.roll(state.history_obs, 1, axis=0).at[0].set(obs)
        updated_act = jnp.roll(state.history_act, 1, axis=0).at[0].set(act)

        # Compute fresh history (reset case)
        state_mid = (self.state_high + self.state_low) / 2
        fresh_obs = jnp.tile(state_mid, (self.n_past, 1))
        fresh_act = jnp.zeros((self.n_past, self.act_dim))

        # If done, use fresh history for next step; otherwise use updated
        new_history_obs = updated_obs * (1 - done) + fresh_obs * done
        new_history_act = updated_act * (1 - done) + fresh_act * done

        return aug_obs, AugmenterState(
            history_obs=new_history_obs, history_act=new_history_act
        )

    def seq(
        self,
        obs_seq: Float[Array, "T obs_dim"],
        act_seq: Float[Array, "T act_dim"],
        done_seq: Float[Array, "T"],
        init_state: AugmenterState,
    ) -> Float[Array, "T aug_dim"]:
        """Augment sequence of observations (raw, no normalization)."""

        def step_fn(state, inputs):
            obs, act, done = inputs
            aug_raw = self._augment_raw(obs, state.history_obs, state.history_act)

            updated_obs = jnp.roll(state.history_obs, 1, axis=0).at[0].set(obs)
            updated_act = jnp.roll(state.history_act, 1, axis=0).at[0].set(act)
            state_mid = (self.state_high + self.state_low) / 2
            fresh_obs = jnp.tile(state_mid, (self.n_past, 1))
            fresh_act = jnp.zeros((self.n_past, self.act_dim))
            new_history_obs = updated_obs * (1 - done) + fresh_obs * done
            new_history_act = updated_act * (1 - done) + fresh_act * done

            return AugmenterState(
                history_obs=new_history_obs, history_act=new_history_act
            ), aug_raw

        _, aug_obs_seq = jax.lax.scan(step_fn, init_state, (obs_seq, act_seq, done_seq))
        return aug_obs_seq


# =============================================================================
# Kernel Augmenter
# =============================================================================


@chex.dataclass
class KernelAugmenterState:
    """State for step-by-step kernel augmentation."""

    history_obs: Float[Array, "n_past obs_dim"]
    history_act: Float[Array, "n_past act_dim"]


@chex.dataclass(frozen=True)
class KernelAugmenter:
    """Kernel-based Augmenter using RBF kernel against reference points.

    Simple augmentation pipeline (before kernel):
    1. state (normalized to [-1, 1])
    2. state_history (normalized) (if use_state_regressor)
    3. action_history (if use_action_regressor)
    4. bias (if add_bias)
    5. co_product: (state @ state.T).ravel() (if use_co_product)

    Then compute RBF kernel against reference points:
    - Reference points are augmented observations from dataset (n_ref points)
    - Features are mean/std normalized before kernel computation
    - Output: exp(-gamma * ||aug_obs_norm - ref_i_norm||^2) for each reference

    Final feature dimension = n_ref (number of kernel values)
    """

    # Reference points (augmented and normalized)
    ref_points: Float[Array, "n_ref simple_aug_dim"]

    # Normalization stats for simple augmented features
    aug_mean: Float[Array, "simple_aug_dim"]
    aug_std: Float[Array, "simple_aug_dim"]

    # RBF kernel parameter
    rbf_gamma: float  # 1 / (2 * sigma^2)

    # Normalization bounds for raw observations
    state_low: Float[Array, "obs_dim"]
    state_high: Float[Array, "obs_dim"]

    # Dimensions and config (static for JIT)
    obs_dim: int
    act_dim: int
    n_past: int
    n_ref: int
    scale_factor: float
    add_bias: bool
    use_state_regressor: bool
    use_action_regressor: bool
    use_co_product: bool

    @staticmethod
    def create(
        obs_dim: int,
        act_dim: int,
        state_low: ArrayLike,
        state_high: ArrayLike,
        ref_dataset: dict,
        n_ref: int = 5000,
        n_past: int = 4,
        scale_factor: float = 1.0,
        add_bias: bool = True,
        use_state_regressor: bool = True,
        use_action_regressor: bool = True,
        use_co_product: bool = True,
        rbf_gamma: float = 1.0,
        seed: int = 42,
    ) -> "KernelAugmenter":
        """Create KernelAugmenter from reference dataset.

        Args:
            ref_dataset: Dict with 'observations', 'actions', 'terminals', 'timeouts'
            n_ref: Number of reference points to sample from dataset
            rbf_gamma: RBF kernel parameter (1 / (2 * sigma^2))
        """
        state_low_arr = jnp.asarray(state_low)
        state_high_arr = jnp.asarray(state_high)

        # Extract data from dataset
        obs_data = jnp.asarray(ref_dataset["observations"])
        act_data = jnp.asarray(ref_dataset["actions"])
        terminals = ref_dataset.get("terminals", jnp.zeros(len(obs_data)))
        timeouts = ref_dataset.get("timeouts", jnp.zeros(len(obs_data)))
        dones = jnp.asarray(terminals) | jnp.asarray(timeouts)

        # Use first n_ref samples (skip first n_past to have valid history)
        sampled_idx = jnp.arange(n_past, n_past + n_ref)

        # Build reference observations and histories
        ref_obs = obs_data[sampled_idx]

        # Build history for each reference point
        def get_history(idx):
            hist_obs = jax.lax.dynamic_slice(
                obs_data, (idx - n_past, 0), (n_past, obs_dim)
            )
            hist_act = jax.lax.dynamic_slice(
                act_data, (idx - n_past, 0), (n_past, act_dim)
            )
            return hist_obs, hist_act

        ref_history_obs, ref_history_act = jax.vmap(get_history)(sampled_idx)

        # Compute simple_aug_dim
        simple_aug_dim = obs_dim
        if use_state_regressor:
            simple_aug_dim += n_past * obs_dim
        if use_action_regressor:
            simple_aug_dim += n_past * act_dim
        if add_bias:
            simple_aug_dim += 1
        if use_co_product:
            simple_aug_dim += obs_dim * obs_dim

        # Create temporary augmenter to compute simple augmentations
        temp = KernelAugmenter(
            ref_points=jnp.zeros((1, simple_aug_dim)),
            aug_mean=jnp.zeros((simple_aug_dim,)),
            aug_std=jnp.ones((simple_aug_dim,)),
            rbf_gamma=1.0,
            state_low=state_low_arr,
            state_high=state_high_arr,
            obs_dim=obs_dim,
            act_dim=act_dim,
            n_past=n_past,
            n_ref=1,
            scale_factor=float(scale_factor),
            add_bias=bool(add_bias),
            use_state_regressor=bool(use_state_regressor),
            use_action_regressor=bool(use_action_regressor),
            use_co_product=bool(use_co_product),
        )

        # Compute simple augmentation for all reference points
        aug_ref = jax.vmap(temp._simple_augment)(
            ref_obs, ref_history_obs, ref_history_act
        )

        # Compute mean/std for normalization
        aug_mean = jnp.mean(aug_ref, axis=0)
        aug_std = jnp.std(aug_ref, axis=0) + 1e-8

        # Normalize reference points
        ref_points_norm = (aug_ref - aug_mean) / aug_std

        return KernelAugmenter(
            ref_points=ref_points_norm,
            aug_mean=aug_mean,
            aug_std=aug_std,
            rbf_gamma=float(rbf_gamma),
            state_low=state_low_arr,
            state_high=state_high_arr,
            obs_dim=obs_dim,
            act_dim=act_dim,
            n_past=n_past,
            n_ref=n_ref,
            scale_factor=float(scale_factor),
            add_bias=bool(add_bias),
            use_state_regressor=bool(use_state_regressor),
            use_action_regressor=bool(use_action_regressor),
            use_co_product=bool(use_co_product),
        )

    def reset(self) -> KernelAugmenterState:
        """Create initial augmenter state."""
        state_mid = (self.state_high + self.state_low) / 2
        return KernelAugmenterState(
            history_obs=jnp.tile(state_mid, (self.n_past, 1)),
            history_act=jnp.zeros((self.n_past, self.act_dim)),
        )

    def simple_aug_dim(self) -> int:
        """Compute simple augmented state dimension (before kernel)."""
        dim = self.obs_dim
        if self.use_state_regressor:
            dim += self.n_past * self.obs_dim
        if self.use_action_regressor:
            dim += self.n_past * self.act_dim
        if self.add_bias:
            dim += 1
        if self.use_co_product:
            dim += self.obs_dim * self.obs_dim
        return dim

    def aug_dim(self) -> int:
        """Final augmented dimension = simple_aug_dim (kernel-weighted sum)."""
        return self.simple_aug_dim()

    def _normalize_obs(
        self, obs: Float[Array, "... obs_dim"]
    ) -> Float[Array, "... obs_dim"]:
        """Normalize observation to [-1, 1]."""
        state_range = self.state_high - self.state_low
        return (obs - self.state_low) / state_range * 2 - 1

    def _simple_augment(
        self,
        obs: Float[Array, "obs_dim"],
        history_obs: Float[Array, "n_past obs_dim"],
        history_act: Float[Array, "n_past act_dim"],
    ) -> Float[Array, "simple_aug_dim"]:
        """Simple augmentation without kernel (for building reference)."""
        obs_norm = self._normalize_obs(obs)
        obs_scaled = obs_norm * self.scale_factor
        scale_sq = self.scale_factor**2
        empty = jnp.zeros((0,))

        return jnp.concatenate(
            (
                obs_scaled,
                self._normalize_obs(history_obs).ravel() * self.scale_factor
                if self.use_state_regressor
                else empty,
                history_act.ravel() if self.use_action_regressor else empty,
                jnp.ones((1,)) if self.add_bias else empty,
                (obs_norm[:, None] * obs_norm[None, :]).ravel() * scale_sq
                if self.use_co_product
                else empty,
            )
        )

    def _augment(
        self,
        obs: Float[Array, "obs_dim"],
        history_obs: Float[Array, "n_past obs_dim"],
        history_act: Float[Array, "n_past act_dim"],
    ) -> Float[Array, "simple_aug_dim"]:
        """Full augmentation: simple augment -> normalize -> RBF kernel weighted sum."""
        aug = self._simple_augment(obs, history_obs, history_act)
        aug_norm = (aug - self.aug_mean) / self.aug_std
        diff = aug_norm - self.ref_points  # (n_ref, simple_aug_dim)
        sq_dist = jnp.sum(diff**2, axis=1)  # (n_ref,)
        weights = jnp.exp(-self.rbf_gamma * sq_dist)  # (n_ref,)
        # Weighted sum of reference features
        return jnp.sum(weights[:, None] * self.ref_points, axis=0)  # (simple_aug_dim,)

    def step(
        self,
        state: KernelAugmenterState,
        obs: Float[Array, "obs_dim"],
        act: Float[Array, "act_dim"],
        done: Float[Array, ""],
    ) -> Tuple[Float[Array, "n_ref"], KernelAugmenterState]:
        """Single step augmentation with history update."""
        aug_obs = self._augment(obs, state.history_obs, state.history_act)

        updated_obs = jnp.roll(state.history_obs, 1, axis=0).at[0].set(obs)
        updated_act = jnp.roll(state.history_act, 1, axis=0).at[0].set(act)

        state_mid = (self.state_high + self.state_low) / 2
        fresh_obs = jnp.tile(state_mid, (self.n_past, 1))
        fresh_act = jnp.zeros((self.n_past, self.act_dim))

        new_history_obs = updated_obs * (1 - done) + fresh_obs * done
        new_history_act = updated_act * (1 - done) + fresh_act * done

        return aug_obs, KernelAugmenterState(
            history_obs=new_history_obs, history_act=new_history_act
        )

    def seq(
        self,
        obs_seq: Float[Array, "T obs_dim"],
        act_seq: Float[Array, "T act_dim"],
        done_seq: Float[Array, "T"],
        init_state: KernelAugmenterState,
    ) -> Float[Array, "T n_ref"]:
        """Augment sequence of observations."""

        def step_fn(state, inputs):
            obs, act, done = inputs
            aug_obs, new_state = self.step(state, obs, act, done)
            return new_state, aug_obs

        _, aug_obs_seq = jax.lax.scan(step_fn, init_state, (obs_seq, act_seq, done_seq))
        return aug_obs_seq


# =============================================================================
# IO Model
# =============================================================================


@chex.dataclass
class IOState:
    """IO model state."""

    theta_su: Float[Array, "aug_dim act_dim"]
    opt_state: optax.OptState
    key: ArrayLike
    norm_state: AugNormState


def io_init(
    aug_dim: int,
    act_dim: int,
    key: ArrayLike,
    optimizer: optax.GradientTransformation,
) -> IOState:
    """Initialize IO model state."""
    key, init_key = jrd.split(key)
    theta_su = jrd.normal(init_key, (aug_dim, act_dim)) * 0.01
    opt_state = optimizer.init(theta_su)
    norm_state = AugNormState.create(aug_dim)
    return IOState(
        theta_su=theta_su, opt_state=opt_state, key=key, norm_state=norm_state
    )


def compute_action(
    theta_su: Float[Array, "aug_dim act_dim"],
    aug_obs: Float[Array, "... aug_dim"],
) -> Float[Array, "... act_dim"]:
    """Compute action: a* = clip(-aug_obs @ theta_su, -1, 1)"""
    return jnp.clip(-jnp.einsum("...s,sa->...a", aug_obs, theta_su), -1.0, 1.0)


def _q_value_batch(
    theta_su: Float[Array, "aug_dim act_dim"],
    aug_obs: Float[Array, "... aug_dim"],
    actions: Float[Array, "... act_dim"],
) -> Float[Array, "..."]:
    """Q(s,a) = ||a||^2 + 2 * s @ theta_su @ a"""
    return jnp.sum(actions**2, axis=-1) + 2 * jnp.sum(
        jnp.einsum("...s,sa->...a", aug_obs, theta_su) * actions, axis=-1
    )


def _loss_fn(
    theta_su: Float[Array, "aug_dim act_dim"],
    aug_obs: Float[Array, "... aug_dim"],
    expert_actions: Float[Array, "... act_dim"],
) -> Float[Array, ""]:
    """Loss: mean(Q(expert) - Q(min))"""
    min_actions = jax.lax.stop_gradient(compute_action(theta_su, aug_obs))
    q_expert = _q_value_batch(theta_su, aug_obs, expert_actions)
    q_min = _q_value_batch(theta_su, aug_obs, min_actions)
    return jnp.mean(q_expert - q_min)


def train_epoch(
    io_state: IOState,
    obs: Float[Array, "N obs_dim"],
    actions: Float[Array, "N act_dim"],
    dones: Float[Array, "N"],
    augmenter: Augmenter,
    optimizer: optax.GradientTransformation,
    seq_len: int,
    batch_size: int,
    step_per_epoch: int,
) -> Tuple[IOState, Float[Array, ""]]:
    """Train for one epoch. Each step uses batch_size sequences of seq_len."""
    n_samples = obs.shape[0]
    max_start = n_samples - seq_len
    key, perm_key = jrd.split(io_state.key)

    start_indices = jrd.randint(perm_key, (step_per_epoch, batch_size), 0, max_start)
    offsets = jnp.arange(seq_len)
    all_indices = start_indices[:, :, None] + offsets[None, None, :]

    init_state = augmenter.reset()
    aug_fn = jax.vmap(augmenter.seq, in_axes=(0, 0, 0, None))

    def batch_step(carry, indices):
        theta_su, opt_state, norm_state = carry
        obs_batch = obs[indices]
        act_batch = actions[indices]
        done_batch = dones[indices]

        aug_raw = aug_fn(obs_batch, act_batch, done_batch, init_state)
        norm_state = update_norm_state(norm_state, aug_raw, augmenter.norm_momentum)
        aug_obs = normalize_aug(aug_raw, norm_state)

        loss, grads = jax.value_and_grad(_loss_fn)(theta_su, aug_obs, act_batch)
        updates, opt_state = optimizer.update(grads, opt_state, theta_su)
        return (optax.apply_updates(theta_su, updates), opt_state, norm_state), loss

    (theta_su, opt_state, norm_state), losses = jax.lax.scan(
        batch_step,
        (io_state.theta_su, io_state.opt_state, io_state.norm_state),
        all_indices,
    )

    return IOState(
        theta_su=theta_su, opt_state=opt_state, key=key, norm_state=norm_state
    ), jnp.mean(losses)
