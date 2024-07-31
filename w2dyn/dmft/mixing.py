"""Module for mixing functionality.

Provides classes that calculate new proposal/trial values from the history of
previously seen results in ways that accelerate the convergence of the loop to
its fixed point.

Apart from classes implementing different mixing algorithms,
- LinearMixer for linear mixing of the previous result
- DiisMixer for Pulay mixing/DIIS of the result history
- NoMixingMixer for no mixing
there are also the decorator classes InitialMixingDecorator, which
allows switching mixers after a number of iterations, as well as
FlatMixingDecorator and RealMixingDecorator, which resp. flatten the
input and separate its real and imaginary parts before passing it on
to the decorated class, which mainly serves the purpose of bringing
more convoluted input quantities into a form suitable for use with
DiisMixer.

All mixer objects are callable, taking the new values as argument and
returning the next proposal/trial value.
"""
import numpy as np

import w2dyn.auxiliaries.deepflatten as deepflatten

class InitialMixingDecorator(object):
    """
    This mixing decorator is switching from the `init_mixer` to the `mixer`
    after `init_count` calls. Useful if `mixer` has unwanted behavior at the
    beginning.
    """
    def __init__(self, init_count, init_mixer, mixer):
        self.init_count = init_count
        self.init_mixer = init_mixer
        self.mixer = mixer
    def __call__(self, *args, custom_residual=None):
        if self.init_count > 0:
            self.init_count -= 1
            return self.init_mixer(*args, custom_residual=custom_residual)
        return self.mixer(*args, custom_residual=custom_residual)

class FlatMixingDecorator(object):
    """
    This mixing decorator takes any kind of nested python lists with numbers and
    numpy arrays and calls `mixer` with a one dimensional copy of the data. The
    shape is restored after `mixer` returned.
    """
    def __init__(self, mixer):
        self.mixer = mixer
    def __call__(self, *args, custom_residual=None):
        if len(args) == 1:
            args = args[0]
        types = deepflatten.types(args)
        shape = deepflatten.shapes(args)
        if custom_residual is not None:
            custom_types = deepflatten.types(custom_residual[1])
            custom_shape = deepflatten.shapes(custom_residual[1])
            custom_residual = (deepflatten.flatten(custom_residual[0])
                               if custom_residual[0] is not None
                               else None,
                               deepflatten.flatten(custom_residual[1]))
        x = deepflatten.flatten(args)
        x = self.mixer(x, custom_residual=custom_residual)
        if custom_residual is not None:
            custom_result = deepflatten.restore(x[1],
                                                custom_shape,
                                                custom_types)
            x = x[0]
        else:
            custom_result = None
        x = deepflatten.restore(x, shape, types)
        return (x if custom_result is None else (x, custom_result))

class RealMixingDecorator(object):
    """
    This mixing decorator takes an array or list as input and concatenates its
    real and imaginary parts to call `mixer`. A numpy array with complex numbers
    is retruned.
    """
    def __init__(self, mixer):
        self.mixer = mixer
    def __call__(self, x, custom_residual=None):
        n = x.shape[0]
        x = np.concatenate([np.real(x), np.imag(x)])
        if custom_residual is not None:
            nc = custom_residual[1].shape[0]
            custom_residual = (np.concatenate([np.real(custom_residual[0]),
                                               np.imag(custom_residual[0])])
                               if custom_residual[0] is not None
                               else None,
                               np.concatenate([np.real(custom_residual[1]),
                                               np.imag(custom_residual[1])]))
        x = self.mixer(x, custom_residual=custom_residual)
        if custom_residual is not None:
            custom_result = x[1][:nc] + 1j * x[1][nc:]
            x = x[0]
        else:
            custom_result = None
        x = x[:n] + 1j*x[n:]
        return (x if custom_result is None else (x, custom_result))

class NoMixingMixer(object):
    """
    This mixer is just passing the input through and therefore not mixing at
    all.
    """
    def __call__(self, *args, custom_residual=None):
        return (args if custom_residual is None else (args, custom_residual[0]))

class LinearMixer(object):
    """
    Allows (linear) under relaxation of quantities in the DMFT loop.

    This is achieved by mixing into the new self-energy a certain share of the
    previous self-energy (controlled by `oldshare`) every time that `mix()` is
    called:

            A^{mixed}_n = (1-oldshare) * A_n + oldshare * A^{mixed}_{n-1}

    thereby exponentially decreasing the influence of the old iterations by
    `\exp(-n \log oldshare)`. This strategy dampens strong statistical
    fluctuations in the QMC solver and ill-defined chemical potentials in
    insulating cases.
    """
    def __init__(self, old_share=0):
        self.old_share = float(old_share)
        self.old_value = None

    def __call__(self, new_value, custom_residual=None):
        if custom_residual is not None:
            raise ValueError()
        if self.old_value is None:
            new_trial = new_value
        else:
            new_trial = self.old_share * self.old_value + (1 - self.old_share) * new_value
        self.old_value = new_trial
        return new_trial

class ModifiedBroydenMixer(object):
    """Modified Broyden mixing, see

    [1] R. Žitko, Convergence Acceleration and Stabilization of
    Dynamical Mean-Field Theory Calculations, Phys. Rev. B 80, 125125
    (2009).

    [2] V. Eyert, A Comparative Study on Methods for Convergence
    Acceleration of Iterative Vector Sequences, Journal of
    Computational Physics 124, 271 (1996).

    [3] D. D. Johnson, Modified Broyden’s Method for Accelerating
    Convergence in Self-Consistent Calculations, Phys. Rev. B 38,
    12807 (1988).

    [4] G. P. Srivastava, Broyden’s Method for Self-Consistent Field
    Convergence Acceleration, J. Phys. A: Math. Gen. 17, L317 (1984).

    [5] D. Vanderbilt and S. G. Louie, Total Energies of Diamond (111)
    Surface Reconstructions by a Linear Combination of Atomic Orbitals
    Method, Phys. Rev. B 30, 6118 (1984).

    """
    def __init__(self, old_share, history, delay, delaylastavg, histkept):
        self.alpha = 1 - old_share

        self.trials = None  # input vectors V^(i)
        self.residuals = None  # fixed-point function results F^(i) = F(V^(i)) != 0
        self.history = history
        self.delay = delay
        self.delaycount = 0
        self.delaylastavg = delaylastavg
        self.histkept = histkept

    def __call__(self, new_value, custom_residual=None):
        history = self.history + self.delaycount
        histkept = history if history > self.histkept else self.histkept

        if self.trials is None:
            # no history yet
            if custom_residual is None:
                new_trial = new_value
            else:
                nc = custom_residual[1].shape[0]
                new_trial = np.concatenate((new_value, custom_residual[0]))
        else:
            if custom_residual is not None:
                nc = custom_residual[1].shape[0]
                if custom_residual[0] is not None:
                    # overwrite stored trial (last proposal) if provided
                    self.trials[-1, -nc:] = custom_residual[0]
            else:
                nc = 0
            trial = self.trials[-1]  # V^(m)
            residual = new_value - trial[:trial.shape[0] - nc]  # F^(m)
            if custom_residual is not None:
                residual = np.concatenate((residual, custom_residual[1]))
                
            if self.residuals is None:
                self.residuals = residual.copy()[np.newaxis, :]
            else:
                self.residuals = np.append(self.residuals, residual[np.newaxis, :], axis=0)

            if self.residuals.shape[0] > histkept:
                self.residuals = self.residuals[-histkept:, ...]
                self.trials = self.trials[-histkept:, ...]

            if self.delay > 0:
                if self.delaycount >= self.delay:
                    trial = self.trials[-(self.delaycount + 1)]
                    if self.delaylastavg <= 1:
                        residual = new_value - trial[:trial.shape[0] - nc]
                        if custom_residual is not None:
                            residual = np.concatenate((residual, custom_residual[1]))
                    else:
                        residual = np.mean(
                            np.concatenate((
                                self.residuals[-self.delaylastavg:, :trial.shape[0] - nc]
                                + self.trials[-self.delaylastavg:, :trial.shape[0] - nc]
                                - trial[np.newaxis, :trial.shape[0] - nc],
                                self.residuals[-self.delaylastavg:, trial.shape[0] - nc:]
                            ), axis=1),
                            axis=0
                        )
                    self.residuals[-(self.delaycount + 1), ...] = residual
                    self.trials = self.trials[:-self.delaycount, ...]
                    self.residuals = self.residuals[:-self.delaycount, ...]

            self.delaycount = (self.delaycount + 1) % (self.delay + 1)

            if self.trials.shape[0] <= 1:
                # linear mixing (custom residual needs to be usable in this way)
                new_trial = trial + self.alpha * residual
            else:
                deltaF = np.diff(self.residuals[-history:, ...], axis=0)
                normDeltaF = np.linalg.norm(deltaF, axis=1, keepdims=True)
                deltaF /= normDeltaF
                deltaV = np.diff(self.trials[-history:, ...], axis=0)/normDeltaF

                # first calculate A, then overwrite it with beta
                beta = np.matmul(deltaF, np.conj(np.transpose(deltaF)))
                beta = np.linalg.pinv((0.001 * np.eye(beta.shape[0]) + beta), hermitian=True)

                new_trial = trial + self.alpha * residual \
                    - np.linalg.multi_dot((residual,
                                           np.conj(np.transpose(deltaF)),
                                           beta,
                                           self.alpha * deltaF + deltaV))

        if self.trials is None:
            self.trials = new_trial.copy()[np.newaxis, :]
        else:
            self.trials = np.append(self.trials, new_trial[np.newaxis, :], axis=0)

        if self.delaycount > 0:
            new_trial[-nc:] = self.trials[-(self.delaycount + 1), -nc:]
        if new_trial.shape != new_value.shape:
            custom_result = new_trial[-nc:]
            new_trial = new_trial[:-nc]
        return new_trial if custom_residual is None else (new_trial, custom_result)


class DiisMixer(object):
    """
    This mixing algorithm, also known as Pulay mixing, uses multiple trials and
    results to combine a (hopefully) better trial for the next iteration.
    
    Futher reading
     - https://en.wikipedia.org/wiki/DIIS
     - P. Pulay, Chem. Phys. Lett. 73, 393
       "Convergence acceleration of iterative sequences.
        The case of SCF iteration"
       https://dx.doi.org/10.1016/0009-2614(80)80396-4
     - P. Pulay, J. Comp. Chem. 3, 556
       "Improved SCF convergence acceleration"
       https://dx.doi.org/10.1002/jcc.540030413
     - P. P. Pratapa, P. Suryanarayana, Chem. Phys. Lett. 635, 69
       "Restarted Pulay mixing for efficient and robust acceleration of
        fixed-point iterations"
       https://dx.doi.org/10.1016/j.cplett.2015.06.029
     - A. S. Banerjee, P. Suryanarayana, J. E. Pask, Chem. Phys. Lett. 647, 31
       "Periodic Pulay method for robust and efficient convergence acceleration
        of self-consistent field iterations"
       https://dx.doi.org/10.1016/j.cplett.2016.01.033
     - H. F. Walker, P. Ni, SIAM J. Numer. Anal., 49, 1715
       "Anderson Acceleration for Fixed-Point Iterations"
       https://dx.doi.org/10.1137/10078356X
       https://core.ac.uk/display/47187107
    """
    def __init__(self, old_share, history, period):
        self.alpha = 1 - old_share
        self.history = history
        self.period = period

        self.i = 0
        self.trials = []
        self.residuals = []

    def __call__(self, new_value, custom_residual=None):
        if self.i <= 0:
            # no history yet
            if custom_residual is None:
                new_trial = new_value
            else:
                nc = custom_residual[1].shape[0]
                new_trial = np.concatenate((new_value, custom_residual[0]))
        else:
            if custom_residual is not None:
                nc = custom_residual[1].shape[0]
                if custom_residual[0] is not None:
                    # overwrite stored trial (last proposal) if provided
                    self.trials[-1][-nc:] = custom_residual[0]
            else:
                nc = 0
            trial = self.trials[-1]
            residual = new_value - trial[:trial.shape[0] - nc]
            if custom_residual is not None:
                residual = np.concatenate((residual, custom_residual[1]))
            self.residuals.append(residual)

            # trim history
            self.trials = self.trials[-self.history:]
            self.residuals = self.residuals[-self.history:]

            if self.i <= 1 or (self.i % self.period) != 0:
                # linear mixing
                new_trial = trial + self.alpha * residual
            else:
                # Pulay mixing
                R = np.array(self.trials); R = R[1:] - R[:-1]; R = R.T
                F = np.array(self.residuals); F = F[1:] - F[:-1]; F = F.T

                new_trial = trial + self.alpha * residual \
                         - np.linalg.multi_dot([R + self.alpha * F, np.linalg.pinv(F), residual])

        self.i += 1
        self.trials.append(new_trial)

        if custom_residual is not None:
            custom_result = new_trial[-nc:]
            new_trial = new_trial[:-nc]

        return new_trial if custom_residual is None else (new_trial, custom_result)
