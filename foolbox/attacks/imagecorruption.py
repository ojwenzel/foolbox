from functools import wraps, partial
from typing import List, Union, Any, Callable, Optional, Tuple, Iterable
from typing_extensions import final, overload
from warnings import warn
import re
from abc import ABC
from abc import abstractmethod
from random import shuffle

import numpy as np
import torch
from torch.multiprocessing import Pool, Process, set_start_method, Queue
import eagerpy as ep

from eagerpy.tensor import PyTorchTensor, NumPyTensor

from ..devutils import flatten
from ..devutils import atleast_kd

from ..distances import l2, linf

from .base import SeverityAttack, AttackWithDistance
from .base import Criterion
from .base import Model
from .base import T
from .base import get_criterion
from .base import get_is_adversarial
from .base import raise_if_kwargs

import imagecorruptions

# MODULE INFORMATION
__author__ = "Ole Jonas Wenzel"
__credits__ = ["Ole Jonas Wenzel", ""]
__version__ = "0.0.1"
__maintainer__ = "Ole Jonas Wenzel"
__email__ = "olejonaswenzel@gmail.com"
__status__ = "Dev"

# CONSTANTS
SEVERITIES = [1, 2, 3, 4, 5]
CORRUPTIONS = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur',
               'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate',
               'jpeg_compression']


# FUNCTIONS
def validate_corruptions(corruptions: Tuple[str, ...]) -> List[str]:
    """

    Validates the elements in the input list or conformity with the corruptions provided by the package
    imagecorruptions.

    :param corruptions: list of corruptions
    :return: elements of the input list, that are valid
    """
    # PARAMETERS
    all_corruptions = CORRUPTIONS
    valid_corruptions = [corruption for corruption in corruptions if corruption in all_corruptions]

    # check if there is any valid corruption provided
    if len(valid_corruptions) == 0:
        raise ValueError('\'corruptions\' has to be a list containing at least one valid element'
                         ' from {}'.format(all_corruptions))

    # remove all invalid corruptions from
    if len(valid_corruptions) != len(corruptions):
        warn('\'corruptions\' contains illegal values. Using: {}'.format(valid_corruptions))

    return list(valid_corruptions)


def validate_severities(severities: List[int]) -> List[int]:
    """

    Validates the elements in the input list or conformity with the severities provided by the package
    imagecorruptions.

    :param severities: list of severities
    :return: elements of the input list, that are valid
    """
    # PARSE INPUT
    if severities is None:
        return SEVERITIES

    # PARAMETERS
    valid_severities = [severity for severity in severities if severity in SEVERITIES]

    # check if there is any valid severity provided
    if len(valid_severities) == 0:
        raise ValueError('\'severities\' has to be a list containing at least one valid element'
                         ' from {}'.format(SEVERITIES))

    # remove all invalid severities from
    if len(valid_severities) != len(severities):
        warn('\'severities\' contains illegal values. Using: {}'.format(valid_severities))

    return list(valid_severities)


def corruption_name_2_cls_name(corruption_name: str) -> str:
    # convert corruption_name as in package imagecorruptions into class name as in this module
    return ''.join([sub.capitalize() for sub in corruption_name.split('_')])


def cls_name_2_corruption_name(cls_name: str):
    """https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case"""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', cls_name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def get_subset(subset: Tuple[str] = ('common',)) -> List[str]:
    if 'common' in subset:
        return CORRUPTIONS[:15]
    elif 'validation' in subset:
        return CORRUPTIONS[15:]
    elif 'all' in subset or subset is None:
        return CORRUPTIONS
    else:
        return validate_corruptions(subset)


def list_corruption_attack_names(subset: Tuple[str] = ('common',)) -> List[str]:
    """

    Retrieves class names for specified subset of corruption attacks. Note: identifiers 'common', 'all', 'validation'
    take precedence over individual corruption_names

    :param subset: specification of subset to retrieve may contain CORRUPTIONS + ['common', 'all', 'validation']
    :return:
    """
    corruptions = get_subset(subset)
    return [corruption_name_2_cls_name(corruption) for corruption in corruptions]


def image_transpose_decorator(func):
    """

    transposes images from format [channels, width , height] to [width, height, channels] before calling function

    :param func: image manipulation function
    :return: function wrapper that pre- and post-processes func input
    """
    @wraps(func)
    def with_transpose(image: np.ndarray, *args, **kwargs):
        image = image.transpose((1, 2, 0))
        image = func(image, *args, **kwargs)
        image = image.transpose((2, 0, 1))

        return image

    return with_transpose


def image_value_range_decorator(func):
    """

    converts the range of pixel values of an image from format [channels, width , height]
    to [width, height, channels] before calling function

    :param func: image manipulation function
    :return: func wrapper that pre- and post-processes func input
    """
    @wraps(func)
    def with_io_transform(image: np.ndarray, *args, **kwargs):
        image = to_255_image(image)
        image = func(image, *args, **kwargs)
        image = to_0_1_image(image)

        return image

    return with_io_transform


def ep_np_ep_conversion_decorator(func: Callable) -> Callable[[ep.Tensor], ep.Tensor]:

    @wraps(func)
    def ep_np_ep_conversion(image: ep.Tensor, *args, **kwargs) -> ep.Tensor:
        return ep.astensor(func(image.numpy(), *args, **kwargs))

    return ep_np_ep_conversion


# @ep_np_ep_conversion_decorator
@image_value_range_decorator
@image_transpose_decorator
def corrupt(image: np.ndarray, corruption: str, severity: int, seed: Optional[int] = None) -> np.ndarray:

    if severity == 0:
        return image

    if seed is not None:
        np.random.seed(seed)

    corrupted_image = imagecorruptions.corrupt(image, corruption_name=corruption, severity=severity)

    return corrupted_image 


def corrupt_images(images: ep.Tensor, corruption: str, severity: int, num_workers: int = 1, random: bool = False):
    """

    :param images: ep.Tensor
        must have leading batch dimension
    :param corruption: str
    :param severity:
    :param num_workers:
    :return:
    """

    # prepend batch dimensions
    if images.ndim == 3:
        images = images[None, :, :, :]

    if images.ndim != 4:
        raise ValueError('Images must be ndarray of dimensionality 4.')

    corrupt_func = partial(corrupt, corruption=corruption, severity=severity)
    seeds = images.sum(axis=(1, 2, 3,)) + severity
    if random:
        seeds = torch.randint(0, seeds.max()[0], size=seeds.shape)
    seeds = seeds.tolist()

    images = images.numpy()
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
        with Pool(num_workers) as p:
            corrupted_images = p.map(corrupt_func, zip(images, seeds), )

    except RuntimeError:
        warn('RuntimeError when attempting parallel image corruption. Corrupting Sequentially.')
        corrupted_images = []
        for image, seed in images:
            corrupted_images.append(corrupt_func(image, seed))

    # corrupted_images = []
    # for image in images:
    #     corrupted_images.append(corrupt_func(image))

    corrupted_images = [ep.astensor(img) for img in corrupted_images]

    return ep.stack(corrupted_images)


def to_255_image(image: np.ndarray) -> np.ndarray:
    return (image * 255).astype(np.uint8)


def to_0_1_image(image: np.ndarray) -> np.ndarray:
    return (image / 255).astype(np.float32)


# CLASSES
class SeverityAttack(AttackWithDistance):
    """Severity attacks try to find adversarials whose perturbation sizes are given by a set of integers"""

    @abstractmethod
    def run(
        self, model: Model, inputs: T, criterion: Any, *, severity: float, **kwargs: Any
    ) -> T:
        """Runs the attack and returns perturbed inputs.

        The size of the perturbations should be at most severity, but this
        is not guaranteed and the caller should verify this or clip the result.
        """
        ...

    @overload
    def __call__(
        self,
        model: Model,
        inputs: T,
        criterion: Any,
        *,
        epsilons: Sequence[Union[float, None]],
        **kwargs: Any,
    ) -> Tuple[List[T], List[T], T]:
        ...

    @overload  # noqa: F811
    def __call__(
        self,
        model: Model,
        inputs: T,
        criterion: Any,
        *,
        epsilons: Union[float, None],
        **kwargs: Any,
    ) -> Tuple[T, T, T]:
        ...

    @final  # noqa: F811
    def __call__(  # type: ignore
        self,
        model: Model,
        inputs: T,
        criterion: Any,
        *,
        epsilons: Union[Sequence[Union[float, None]], float, None],
        **kwargs: Any,
    ) -> Union[Tuple[List[T], List[T], T], Tuple[T, T, T]]:

        x, restore_type = ep.astensor_(inputs)
        del inputs

        criterion = get_criterion(criterion)
        is_adversarial = get_is_adversarial(criterion, model)

        was_iterable = True
        if not isinstance(epsilons, Iterable):
            epsilons = [epsilons]
            was_iterable = False

        N = len(x)
        K = len(epsilons)

        # None means: just minimize, no early stopping, no limit on the perturbation size
        if any(eps is None for eps in epsilons):
            # TODO: implement a binary search
            raise NotImplementedError(
                "SeverityAttack subclasses do not yet support None in epsilons"
            )
        real_epsilons = [eps for eps in epsilons if eps is not None]
        del epsilons

        xps = []
        xpcs = []
        success = []
        for epsilon in real_epsilons:
            xp = self.run(model, x, criterion, severity=epsilon, **kwargs)

            # clip to severity because we don't really know what the attack returns;
            # alternatively, we could check if the perturbation is at most severity,
            # but then we would need to handle numerical violations;
            xpc = self.distance.clip_perturbation(x, xp, epsilon)
            is_adv = is_adversarial(xp)

            xps.append(xp)
            xpcs.append(xpc)
            success.append(is_adv)

        # # TODO: the correction we apply here should make sure that the limits
        # # are not violated, but this is a hack and we need a better solution
        # # Alternatively, maybe can just enforce the limits in __call__
        # xps = [
        #     self.run(model, x, criterion, severity=severity, **kwargs)
        #     for severity in real_epsilons
        # ]

        # is_adv = ep.stack([is_adversarial(xp) for xp in xps])
        # assert is_adv.shape == (K, N)

        # in_limits = ep.stack(
        #     [
        #         self.distance(x, xp) <= severity
        #         for xp, severity in zip(xps, real_epsilons)
        #     ],
        # )
        # assert in_limits.shape == (K, N)

        # if not in_limits.all():
        #     # TODO handle (numerical) violations
        #     # warn user if run() violated the severity constraint
        #     import pdb

        #     pdb.set_trace()

        # success = ep.logical_and(in_limits, is_adv)
        # assert success.shape == (K, N)

        success_ = ep.stack(success)
        assert success_.shape == (K, N)

        xps_ = [restore_type(xp) for xp in xps]
        xpcs_ = [restore_type(xpc) for xpc in xpcs]

        if was_iterable:
            return xps_, xpcs_, restore_type(success_)
        else:
            assert len(xps_) == 1
            assert len(xpcs_) == 1
            return xps_[0], xpcs_[0], restore_type(success_.squeeze(axis=0))


class BaseImageCorruptionAttack(SeverityAttack, ABC):
    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Criterion, Any] = None,
        *,
        epsilon: int,
        **kwargs: Any,
    ) -> T:
        # raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)

        repeats: int = 1 if not self.stochastic else kwargs.pop('repeats', 1)
        num_workers: int = max([1, kwargs.pop('num_workers', 1)])
        check_trivial: bool = kwargs.pop('check_trivial', False)

        del criterion, kwargs, inputs

        is_adversarial = get_is_adversarial(criterion_, model)

        result = x
        if check_trivial:
            found = is_adversarial(result)
        else:
            found = ep.zeros(x, len(result)).bool()

        for _ in range(repeats):
            if found.all():
                break
            # min_, max_ = model.bounds
            adv = self.sample_imagecorruption_noise(images=x, severity=epsilon, num_workers=num_workers)
            # norms = self.get_norms(adv - x)
            # p = p / atleast_kd(norms, p.ndim)
            # x = x + severity * p
            # x = x.clip(min_, max_)
            # if not isinstance(x, adv.__class__):
            #     if isissh aorakinstance(adv, (NumPyTensor, PyTorchTensor)):
            #         adv_list = adv.raw.tolist()
            #     else:
            #         raise NotImplementedError
            #     adv = type(x)(adv_list)

            adv = restore_type(adv)
            adv = ep.astensor(torch.from_numpy(adv.raw).to(model.device))

            is_adv = is_adversarial(adv)
            is_new_adv = ep.logical_and(is_adv, ep.logical_not(found))
            result = ep.where(atleast_kd(is_new_adv, adv.ndim), adv, result)
            found = ep.logical_or(found, is_adv)

        result = restore_type(result)

        return result

    @abstractmethod
    def sample_imagecorruption_noise(self, x: ep.Tensor, severity: int) -> ep.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_norms(self, p: ep.Tensor) -> ep.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def stochastic(self) -> bool:
        NotImplementedError


class L2Mixin:
    distance = l2

    def get_norms(self, p: ep.Tensor) -> ep.Tensor:
        return flatten(p).norms.l2(axis=-1)


class GaussianNoise(L2Mixin, BaseImageCorruptionAttack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'gaussian_noise'

    sample_imagecorruption_noise = partial(corrupt_images, corruption=CORRUPTION)

    @property
    def stochastic(self) -> bool:
        return True

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class ShotNoise(L2Mixin, BaseImageCorruptionAttack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'shot_noise'

    sample_imagecorruption_noise = partial(corrupt_images, corruption=CORRUPTION)

    @property
    def stochastic(self) -> bool:
        return True

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class ImpulseNoise(L2Mixin, BaseImageCorruptionAttack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'impulse_noise'

    sample_imagecorruption_noise = partial(corrupt_images, corruption=CORRUPTION)

    @property
    def stochastic(self) -> bool:
        return True

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class DefocusBlur(L2Mixin, BaseImageCorruptionAttack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'defocus_blur'

    sample_imagecorruption_noise = partial(corrupt_images, corruption=CORRUPTION)

    @property
    def stochastic(self) -> bool:
        return False

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class GlassBlur(L2Mixin, BaseImageCorruptionAttack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'glass_blur'

    sample_imagecorruption_noise = partial(corrupt_images, corruption=CORRUPTION)

    @property
    def stochastic(self) -> bool:
        return False

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class MotionBlur(L2Mixin, BaseImageCorruptionAttack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'motion_blur'

    sample_imagecorruption_noise = partial(corrupt_images, corruption=CORRUPTION)

    @property
    def stochastic(self) -> bool:
        return True

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class ZoomBlur(L2Mixin, BaseImageCorruptionAttack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'zoom_blur'

    sample_imagecorruption_noise = partial(corrupt_images, corruption=CORRUPTION)

    @property
    def stochastic(self) -> bool:
        return False

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class Snow(L2Mixin, BaseImageCorruptionAttack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'snow'

    sample_imagecorruption_noise = partial(corrupt_images, corruption=CORRUPTION)

    @property
    def stochastic(self) -> bool:
        return True

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class Frost(L2Mixin, BaseImageCorruptionAttack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'frost'

    sample_imagecorruption_noise = partial(corrupt_images, corruption=CORRUPTION)

    @property
    def stochastic(self) -> bool:
        return True

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class Fog(L2Mixin, BaseImageCorruptionAttack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'fog'

    sample_imagecorruption_noise = partial(corrupt_images, corruption=CORRUPTION)

    @property
    def stochastic(self) -> bool:
        return True

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class Brightness(L2Mixin, BaseImageCorruptionAttack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'brightness'

    sample_imagecorruption_noise = partial(corrupt_images, corruption=CORRUPTION)

    @property
    def stochastic(self) -> bool:
        return False

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class Contrast(L2Mixin, BaseImageCorruptionAttack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'contrast'

    sample_imagecorruption_noise = partial(corrupt_images, corruption=CORRUPTION)

    @property
    def stochastic(self) -> bool:
        return False

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class ElasticTransform(L2Mixin, BaseImageCorruptionAttack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'elastic_transform'

    sample_imagecorruption_noise = partial(corrupt_images, corruption=CORRUPTION)

    @property
    def stochastic(self) -> bool:
        return True

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class Pixelate(L2Mixin, BaseImageCorruptionAttack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'pixelate'

    sample_imagecorruption_noise = partial(corrupt_images, corruption=CORRUPTION)

    @property
    def stochastic(self) -> bool:
        return False

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class JpegCompression(L2Mixin, BaseImageCorruptionAttack):
    """Increases the amount of image corruption until the input is misclassified.

    """
    
    CORRUPTION = 'jpeg_compression'

    sample_imagecorruption_noise = partial(corrupt_images, corruption=CORRUPTION)

    @property
    def stochastic(self) -> bool:
        return False

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class RotateImageCorruptionsAttack(L2Mixin, BaseImageCorruptionAttack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    self._attacks: Optional[List[BaseImageCorruptionAttack]] = None

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Criterion, Any] = None,
        *,
        epsilon: int,
        **kwargs: Any,
    ) -> T:
        # raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)

        num_workers: int = max([1, kwargs.pop('num_workers', 1)])
        repeats: int = kwargs.pop('repeats', 1)
        attacks = self.attacks
        shuffle(attacks)
        num_attacks: int = len(attacks)
        check_trivial: bool = kwargs.pop('check_trivial', False)

        del criterion, kwargs, inputs

        is_adversarial = get_is_adversarial(criterion_, model)

        result = x
        if check_trivial:
            found = is_adversarial(result)
        else:
            found = ep.zeros(x, len(result)).bool()

        for _ in range(repeats):
            if found.all():
                break
            # norms = self.get_norms(adv - x)
            # p = p / atleast_kd(norms, p.ndim)
            # x = x + severity * p
            # x = x.clip(min_, max_)
            # if not isinstance(x, adv.__class__):
            #     if isissh aorakinstance(adv, (NumPyTensor, PyTorchTensor)):
            #         adv_list = adv.raw.tolist()
            #     else:
            #         raise NotImplementedError
            #     adv = type(x)(adv_list)

            # ALLOCATION
            advs = ep.zeros_like(x)
            found = ep.astensor(torch.zeros(advs.shape[0], dtype=torch.bool))
            successful_attack = -1 * ep.ones_like(found).int()

            for a, attack in enumerate(attacks):

                if found.all():
                    break

                [adv, _, success] = attack(model,
                                           x[~found],
                                           criterion_,
                                           epsilons=[epsilon],  # always only one severity!
                                           repeats=repeats,
                                           num_workers=num_workers)

                assert success.ndim == 1, f"Expected success to have dimensionality 1, got {success.ndim}"

                # attack those images, that have not been successfully attacked yet
                new_successes = torch.zeros_like(found, dtype=torch.bool)
                new_successes[~found] = success

                advs[new_successes] = torch.cat(adv, dim=0)[success]
                successful_attack[new_successes] = a
                found[new_successes] = True

            advs = restore_type(advs)

            is_adv = is_adversarial(advs)
            is_new_adv = ep.logical_and(is_adv, ep.logical_not(found))
            result = ep.where(atleast_kd(is_new_adv, advs.ndim), advs, result)
            found = ep.logical_or(found, is_adv)

        result = restore_type(result)

        return result

    @property
    def attacks(self) -> List[BaseImageCorruptionAttack]:
        if self._attacks is None:
            self._attacks = get_corruption_attacks('all')
        return self._attacks


def get_corruption_attacks(subset: Union[Tuple[str], str] = ('common',)) -> List[BaseImageCorruptionAttack]:
    """

    Retrieves classes for specified subset of corruption attacks. Note: identifiers 'common', 'all', 'validation' take
    precedence over individual corruption_names

    :param subset: specification of subset to retrieve may contain CORRUPTIONS + ['common', 'all', 'validation']
    :return:
    """

    if isinstance(subset, str):
        subset = (subset, )

    corruptions = list_corruption_attack_names(subset)
    return [globals()[corruption] for corruption in corruptions]
