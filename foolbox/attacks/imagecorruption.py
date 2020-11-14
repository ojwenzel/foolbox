from functools import wraps, partial
from typing import List, Union, Any, Callable, Optional, Tuple, Iterable, Sequence
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

from .base import AttackWithDistance, Attack
from .base import Criterion
from .base import Model
from .base import T
from .base import get_criterion
from .base import get_is_adversarial
from .base import raise_if_kwargs
from ..criteria import TargetedMisclassification

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
def logical_xor(a: ep.Tensor, b: ep.Tensor) -> ep.Tensor:
    not_a, not_b = ep.logical_not(a), ep.logical_not(b)
    return ep.logical_or(ep.logical_and(a, b), ep.logical_and(not_a, not_b))

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
def corrupt(image: np.ndarray, seed: Optional[int], corruption: str, severity: int) -> np.ndarray:

    if severity == 0:
        return image

    if seed is not None:
        np.random.seed(seed)

    corrupted_image = imagecorruptions.corrupt(image, corruption_name=corruption, severity=severity)

    return corrupted_image 


def corrupt_images(images: ep.Tensor,
                   corruption: str,
                   severity: int,
                   num_workers: int = 1,
                   random: bool = True) -> ep.Tensor:
    """

    :param images: ep.Tensor
        must have leading batch dimension
    :param corruption: str
    :param severity:
    :param num_workers:
    :param
    :return:
    """

    # prepend batch dimensions
    if images.ndim == 3:
        images = images[None, :, :, :]

    if images.ndim != 4:
        raise ValueError('Images must be ndarray of dimensionality 4.')

    corrupt_func = partial(corrupt, corruption=corruption, severity=severity)
    seeds = images.sum(axis=(1, 2, 3,)).numpy().astype(int) + severity
    if random:
        seeds = [None] * len(images)

    images, restore_type = ep.astensor_(images)
    np_images = images.numpy()
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
        with Pool(num_workers) as p:
            corrupted_images = p.starmap(corrupt_func, zip(np_images, seeds), )

    except RuntimeError:
        warn('RuntimeError when attempting parallel image corruption. Corrupting Sequentially.')
        corrupted_images = []
        for image, seed in zip(np_images, seeds):
            corrupted_images.append(corrupt_func(image, seed))

    # corrupted_images = []
    # for image in images:
    #     corrupted_images.append(corrupt_func(image))
    corrupted_images = [ep.from_numpy(restore_type(images), img) for img in corrupted_images]

    return ep.stack(corrupted_images)


def to_255_image(image: np.ndarray) -> np.ndarray:
    return (image * 255).astype(np.uint8)


def to_0_1_image(image: np.ndarray) -> np.ndarray:
    return (image / 255).astype(np.float32)


# CLASSES
class IntegerIntensityAttack(ABC):

    @abstractmethod
    def run(
            self, model: Model, inputs: T, criterion: Any, *, severities: Iterable[int], **kwargs: Any
    ) -> Tuple[T, T]:
        """Runs the attack and returns perturbed inputs.

        The size of the perturbations should be at most epsilon, but this
        is not guaranteed and the caller should verify this or clip the result.
        """
        ...

    def __call__(
            self,
            model: Model,
            inputs: T,
            criterion: Any,
            *,
            severities: Union[Sequence[int], None],
            **kwargs: Any,
    ) -> Tuple[T, T, T]:
        x, restore_type = ep.astensor_(inputs)
        del inputs

        criterion = get_criterion(criterion)
        is_adversarial = get_is_adversarial(criterion, model)

        if severities is None:
            raise NotImplementedError(
                "IntegerIntensityAttack subclasses do not yet support severity None"
            )

        xp, distances = self.run(model, x, criterion, severities=severities, **kwargs)
        success = is_adversarial(xp)

        xp_ = restore_type(xp)

        return xp_, restore_type(success), restore_type(distances)

    @abstractmethod
    def repeat(self, times: int) -> "IntegerIntensityAttack":
        ...

    def __repr__(self) -> str:
        args = ", ".join(f"{k.strip('_')}={v}" for k, v in vars(self).items())
        return f"{self.__class__.__name__}({args})"


class RepeatedIntegerIntensityAttack(IntegerIntensityAttack):
    """Repeats the wrapped attack and returns the best result"""

    def __init__(self, attack: IntegerIntensityAttack, times: int):
        if times < 1:
            raise ValueError(f"expected times >= 1, got {times}")  # pragma: no cover

        self.attack: IntegerIntensityAttack = attack
        self.times: int = times

    def __call__(  # noqa: F811
            self,
            model: Model,
            inputs: T,
            criterion: TargetedMisclassification,
            *,
            severities: Union[Sequence[int], int, None],
            check_trivial: bool = True,
            **kwargs: Any,
    ) -> Tuple[T, T, T]:
        x, restore_type = ep.astensor_(inputs)
        del inputs

        target_classes = criterion.target_classes
        criterion_ = get_criterion(criterion)
        is_adversarial = get_is_adversarial(criterion_, model)

        if not isinstance(severities, Iterable):
            severities = [severities]

        N = len(x)
        K = len(severities)

        result = x
        if check_trivial:
            found = is_adversarial(result)
        else:
            found = ep.zeros(x, len(result)).bool()
        result_distance = -1 * ep.where(found, 0., ep.nan)  # 0: trivial adversarial, nan: no adversarial

        for i in range(self.times):

            if found.all():
                break

            # run the attack
            criterion_ = criterion
            criterion_.target_classes = target_classes[ep.logical_not(found)]
            is_adversarial = get_is_adversarial(criterion_, model)
            xps, distance = self.run(
                model, x[ep.logical_not(found)], criterion_, severities=severities, **kwargs
            )

            success = is_adversarial(xps)

            n = ep.logical_not(found).sum()
            assert len(xps) == n
            assert success.shape == (n,)
            assert distance.shape == (n,)
            assert attack.shape == (n,)

            # upscale new successes
            is_adv = np.zeros_like(found, dtype=np.bool)
            is_adv[ep.logical_not(found).numpy()] = success.numpy()
            is_adv = ep.from_numpy(restore_type(found), is_adv)
            is_new_adv = ep.logical_and(is_adv, ep.logical_not(found))

            # upscale adversarial examples
            xps_ = np.zeros_like(x.numpy())
            xps_[is_adv.numpy()] = xps[success.bool()].numpy()
            xps_ = ep.from_numpy(restore_type(found), xps_)

            # upscale new distances
            new_distances = np.ones_like(is_new_adv.numpy(), dtype='float32') * np.nan
            new_distances[ep.logical_not(found).numpy()] = distance.numpy()
            new_distances = ep.from_numpy(found, new_distances)

            result = ep.where(atleast_kd(is_new_adv, xps_.ndim), xps_, result)
            result_distance = ep.where(atleast_kd(is_new_adv, distance.ndim),
                                       new_distances,
                                       result_distance)
            found = ep.logical_or(found, is_adv)

        result = restore_type(result)
        result_distance = restore_type(result_distance)

        return result, restore_type(found), result_distance

    def run(self,
            model: Model,
            inputs: T,
            criterion: TargetedMisclassification,
            *,
            severities: Sequence[int],
            **kwargs) -> Tuple[T, T]:

        adv, _, distance = self.attack(
            model, inputs, criterion, epsilons=severities, **kwargs
        )

        return adv, distance

    def repeat(self, times: int) -> "RepeatedIntegerIntensityAttack":
        return RepeatedIntegerIntensityAttack(self.attack, self.times * times)


class ImageCorruptionAttack(IntegerIntensityAttack, ABC):
    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Criterion, Any] = None,
        *,
        severities: Sequence[int],
        **kwargs: Any,
    ) -> Tuple[T, T]:
        # raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)

        # repeats: int = 1 if not self.stochastic else kwargs.pop('repeats', 1)
        num_workers: int = max([1, kwargs.pop('num_workers', 1)])
        check_trivial: bool = kwargs.pop('check_trivial', False)
        random: bool = kwargs.pop('random', True)

        del criterion, kwargs, inputs

        is_adversarial = get_is_adversarial(criterion_, model)

        result = x
        if check_trivial:
            found = is_adversarial(result)
        else:
            found = ep.zeros(x, len(result)).bool()
        distance = ep.zeros_like(found).astype(int)

        for s, severity in enumerate(severities):

            if found.all():
                break

            adv = self.sample_imagecorruption_noise(images=x[ep.logical_not(found)], severity=severity,
                                                    num_workers=num_workers, random=random)

            is_adv = np.zeros_like(found, dtype=np.bool)
            is_adv[ep.logical_not(found).numpy()] = is_adversarial(adv).numpy()
            is_adv = ep.from_numpy(restore_type(found), is_adv)
            is_new_adv = ep.logical_and(is_adv, ep.logical_not(found))

            result = ep.where(atleast_kd(is_new_adv, adv.ndim), adv, result)
            distance = ep.where(atleast_kd(is_new_adv, distance.ndim), ep.ones_like(distance) * severity, distance)
            found = ep.logical_or(found, is_adv)

        result = restore_type(result)
        distance = restore_type(distance)

        return result, distance

    @abstractmethod
    def sample_imagecorruption_noise(self,
                                     images: ep.Tensor,
                                     severity: int,
                                     num_workers: int = 1,
                                     random: bool = True) -> ep.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def stochastic(self) -> bool:
        pass

    def repeat(self, times: int) -> "IntegerIntensityAttack":
        return RepeatedIntegerIntensityAttack(attack=self, times=times)


class GaussianNoise(ImageCorruptionAttack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'gaussian_noise'

    sample_imagecorruption_noise = partial(corrupt_images, corruption=CORRUPTION)

    @property
    def stochastic(self) -> bool:
        return True

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class ShotNoise(ImageCorruptionAttack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'shot_noise'

    sample_imagecorruption_noise = partial(corrupt_images, corruption=CORRUPTION)

    @property
    def stochastic(self) -> bool:
        return True

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class ImpulseNoise(ImageCorruptionAttack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'impulse_noise'

    sample_imagecorruption_noise = partial(corrupt_images, corruption=CORRUPTION)

    @property
    def stochastic(self) -> bool:
        return True

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class DefocusBlur(ImageCorruptionAttack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'defocus_blur'

    sample_imagecorruption_noise = partial(corrupt_images, corruption=CORRUPTION)

    @property
    def stochastic(self) -> bool:
        return False

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class GlassBlur(ImageCorruptionAttack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'glass_blur'

    sample_imagecorruption_noise = partial(corrupt_images, corruption=CORRUPTION)

    @property
    def stochastic(self) -> bool:
        return False

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class MotionBlur(ImageCorruptionAttack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'motion_blur'

    sample_imagecorruption_noise = partial(corrupt_images, corruption=CORRUPTION)

    @property
    def stochastic(self) -> bool:
        return True

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class ZoomBlur(ImageCorruptionAttack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'zoom_blur'

    sample_imagecorruption_noise = partial(corrupt_images, corruption=CORRUPTION)

    @property
    def stochastic(self) -> bool:
        return False

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class Snow(ImageCorruptionAttack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'snow'

    sample_imagecorruption_noise = partial(corrupt_images, corruption=CORRUPTION)

    @property
    def stochastic(self) -> bool:
        return True

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class Frost(ImageCorruptionAttack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'frost'

    sample_imagecorruption_noise = partial(corrupt_images, corruption=CORRUPTION)

    @property
    def stochastic(self) -> bool:
        return True

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class Fog(ImageCorruptionAttack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'fog'

    sample_imagecorruption_noise = partial(corrupt_images, corruption=CORRUPTION)

    @property
    def stochastic(self) -> bool:
        return True

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class Brightness(ImageCorruptionAttack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'brightness'

    sample_imagecorruption_noise = partial(corrupt_images, corruption=CORRUPTION)

    @property
    def stochastic(self) -> bool:
        return False

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class Contrast(ImageCorruptionAttack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'contrast'

    sample_imagecorruption_noise = partial(corrupt_images, corruption=CORRUPTION)

    @property
    def stochastic(self) -> bool:
        return False

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class ElasticTransform(ImageCorruptionAttack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'elastic_transform'

    sample_imagecorruption_noise = partial(corrupt_images, corruption=CORRUPTION)

    @property
    def stochastic(self) -> bool:
        return True

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class Pixelate(ImageCorruptionAttack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'pixelate'

    sample_imagecorruption_noise = partial(corrupt_images, corruption=CORRUPTION)

    @property
    def stochastic(self) -> bool:
        return False

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class JpegCompression(ImageCorruptionAttack):
    """Increases the amount of image corruption until the input is misclassified.

    """
    
    CORRUPTION = 'jpeg_compression'

    sample_imagecorruption_noise = partial(corrupt_images, corruption=CORRUPTION)

    @property
    def stochastic(self) -> bool:
        return False

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class RotateImageCorruptionsAttack:
    """Increases the amount of image corruption until the input is misclassified.

    """

    def __init__(self, attacks: Tuple[ImageCorruptionAttack]):
        self.attacks: Sequence[ImageCorruptionAttack] = attacks

    def __call__(
            self,
            model: Model,
            inputs: T,
            criterion: TargetedMisclassification,
            *,
            severities: Union[Sequence[int], None],
            attack_permutation: Optional[torch.tensor] = None,
            **kwargs: Any,
    ) -> Tuple[T, T, T, T]:

        x, restore_type = ep.astensor_(inputs)
        target_classes = criterion.target_classes
        kwargs['attack_permutation'] = attack_permutation
        del inputs

        if severities is None:
            raise NotImplementedError(
                "IntegerIntensityAttack subclasses do not yet support severity None"
            )

        xp, success, distances, attacks = self.run(model, x, criterion, severities=severities, **kwargs)

        criterion.target_classes = target_classes
        criterion_ = get_criterion(criterion)
        is_adversarial = get_is_adversarial(criterion_, model)
        assert logical_xor(success, is_adversarial(xp)).all()
        # success = is_adversarial(xp)

        xp_ = restore_type(xp)

        return (xp_,
                restore_type(success),
                restore_type(distances),
                restore_type(attacks), )

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: TargetedMisclassification,
        *,
        severities: Sequence[int],
        **kwargs: Any,
    ) -> Tuple[T, T, T, T]:
        # raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)

        num_workers: int = max([1, kwargs.pop('num_workers', 1)])
        check_trivial: bool = kwargs.pop('check_trivial', False)
        attack_permutation: Optional[torch.tensor] = kwargs.pop('attack_permutation', None)

        del kwargs, inputs

        is_adversarial = get_is_adversarial(criterion_, model)

        result = x
        if check_trivial:
            found = is_adversarial(result)
        else:
            found = ep.zeros(x, len(result)).bool()
        # print(f'input, shape: {result.shape}')
        # print(f'found, shape: {found.shape}, number found: {found.sum().item()}')
        result_attack = -1 * ep.where(found, -1., ep.nan)  # -1: trivial adversarial, nan: no adversarial
        result_distance = -1 * ep.where(found, 0., ep.nan)  # 0: trivial adversarial, nan: no adversarial
        target_classes = criterion.target_classes[ep.logical_not(found)]

        # shuffle attacks for each image
        num_attacks = len(self.attacks)
        # print(f'attack permutation, shape: {attack_permutation.shape}')
        attacks = attack_permutation
        if attack_permutation is None:
            attacks = ep.from_numpy(restore_type(found),
                                    np.stack([np.random.permutation(num_attacks) for _ in range(len(x))]))

        for s, severity in enumerate(severities):

            if found.all():
                break

            for a_i in range(num_attacks):

                if found.all():
                    break

                # select attack index for each image
                attack_idx = attacks[:, a_i]

                for a_j in range(num_attacks):

                    if found.all():
                        break

                    attack = self.attacks[a_j]

                    # select images for which no adversarial has been found yet and shall be attacked with attack a_j, now
                    # print(f'a_i: {a_i}', f'a_j: {a_j}\n',
                    #       f'severity: {severity}\n',
                    #       f'attack selection, shape: {(attack_idx == a_j).shape}, selected: {(attack_idx == a_j).sum().item()}\n',
                    #       f'not found, shape: {ep.logical_not(found).shape}, number not found: {ep.logical_not(found).sum().item()}')
                    selection = ep.logical_and(attack_idx == a_j, ep.logical_not(found))

                    criterion.target_classes = target_classes[selection]
                    [adv, _, success] = attack(model,
                                               x[selection],
                                               criterion,
                                               severities=[severity],  # always only one severity!
                                               num_workers=num_workers)

                    assert success.ndim == 1, f"Expected success to have dimensionality 1, got {success.ndim}"

                    adv = restore_type(adv)

                    # Use numpy to map currently, sucessfully attacked images back to the set of all images.
                    # Since ep.Tensor does not allow assignment, we perform the mapping with numpy tensors and convert
                    # back to ep tensors thereafter.
                    is_adv = np.zeros_like(found, dtype=np.bool)
                    is_adv[selection.numpy()] = success.bool().numpy()
                    is_adv = ep.from_numpy(restore_type(found), is_adv)
                    is_new_adv = ep.logical_and(is_adv, ep.logical_not(found))

                    advs = np.zeros_like(x.numpy())
                    advs[is_adv.numpy()] = adv[success.bool()].numpy()
                    advs = ep.from_numpy(restore_type(found), advs)

                    result = ep.where(atleast_kd(is_new_adv, advs.ndim), advs, result)
                    result_attack = ep.where(atleast_kd(is_new_adv, result_attack.ndim),
                                             ep.ones_like(result_attack) * a_j, result_attack)
                    result_distance = ep.where(atleast_kd(is_new_adv, result_distance.ndim),
                                               ep.ones_like(result_distance) * severity, result_distance)
                    found = ep.logical_or(found, is_adv)

        result_attack = ep.where(found, result_attack, ep.nan * ep.ones_like(result_attack))
        result_distance = ep.where(found, result_distance, ep.nan * ep.ones_like(result_distance))

        result = restore_type(result)

        return result, restore_type(found), restore_type(result_distance), restore_type(result_attack)

    @abstractmethod
    def repeat(self, times: int) -> "RepeatedRotateImageCorruptionsAttack":
        return RepeatedRotateImageCorruptionsAttack(attack=self, times=times)

    def __repr__(self) -> str:
        args = ", ".join(f"{k.strip('_')}={v}" for k, v in vars(self).items())
        return f"{self.__class__.__name__}({args})"


class RepeatedRotateImageCorruptionsAttack:
    """Repeats the wrapped attack and returns the best result"""

    def __init__(self, attack: RotateImageCorruptionsAttack, times: int):
        if times < 1:
            raise ValueError(f"expected times >= 1, got {times}")  # pragma: no cover

        self.attack: RotateImageCorruptionsAttack = attack
        self.times: int = times

    def __call__(  # noqa: F811
            self,
            model: Model,
            inputs: T,
            criterion: TargetedMisclassification,
            *,
            severities: Union[Sequence[int], int, None],
            check_trivial: bool = True,
            **kwargs: Any,
    ) -> Tuple[T, T, T, T]:
        x, restore_type = ep.astensor_(inputs)
        seed: Optional[int] = kwargs.pop('seed', None)

        del inputs

        target_classes = criterion.target_classes
        criterion_ = get_criterion(criterion)
        is_adversarial = get_is_adversarial(criterion_, model)

        if seed is not None:
            np.random.seed(seed)

        if not isinstance(severities, Iterable):
            severities = [severities]

        N = len(x)
        K = len(severities)

        result = x
        if check_trivial:
            found = is_adversarial(result)
        else:
            found = ep.zeros(x, len(result)).bool()
        result_distance = -1 * ep.where(found, 0., ep.nan)  # 0: trivial adversarial, nan: no adversarial
        result_attack = -1 * ep.where(found, -1., ep.nan)  # -1: trivial adversarial, nan: no adversarial

        num_attacks = len(self.attack.attacks)
        attack_permutation = ep.from_numpy(found,
                                           np.stack([np.random.permutation(num_attacks) for _ in range(len(x))]))
        kwargs['attack_permutation'] = attack_permutation

        for i in range(self.times):

            if found.all():
                break

            not_found = ep.logical_not(found)

            # run the attack
            criterion_ = criterion
            criterion_.target_classes = target_classes[not_found]
            is_adversarial = get_is_adversarial(criterion_, model)
            kwargs['attack_permutation'] = attack_permutation[not_found, :]
            xps, success, distance, attack = self.run(
                model,
                x[not_found],
                criterion_, severities=severities, **kwargs
            )
            assert logical_xor(is_adversarial(xps), success).all()

            n = ep.logical_not(found).sum()
            assert len(xps) == n
            assert success.shape == (n,)
            assert distance.shape == (n,)
            assert attack.shape == (n,)

            # upscale new successes
            is_adv = np.zeros_like(found, dtype=np.bool)
            is_adv[not_found.numpy()] = success.bool().numpy()
            is_adv = ep.from_numpy(found, is_adv)
            is_new_adv = ep.logical_and(is_adv, not_found)
            
            # upscale adversarial examples
            xps_ = np.zeros_like(x.numpy())
            xps_[is_adv.numpy()] = xps[success.bool()].numpy()
            xps_ = ep.from_numpy(found, xps_)

            # upscale new distances
            new_distances = np.ones_like(is_new_adv.numpy(), dtype='float32') * np.nan
            new_distances[not_found.numpy()] = distance.numpy()
            new_distances = ep.from_numpy(found, new_distances)

            # upscale new attacks
            new_attacks = np.ones_like(is_new_adv.numpy(), dtype='float32') * np.nan
            new_attacks[not_found.numpy()] = attack.numpy()
            new_attacks = ep.from_numpy(found, new_attacks)

            result = ep.where(atleast_kd(is_new_adv, xps_.ndim), xps_, result)
            result_distance = ep.where(atleast_kd(is_new_adv, distance.ndim),
                                       new_distances,
                                       result_distance)
            result_attack = ep.where(atleast_kd(is_new_adv, attack.ndim),
                                     new_attacks,
                                     result_attack)
            found = ep.logical_or(found, is_adv)

        result = restore_type(result)
        result_distance = restore_type(result_distance)
        result_attack = restore_type(result_attack)

        return result, restore_type(found), result_distance, result_attack

    def run(self,
            model: Model,
            inputs: T,
            criterion: TargetedMisclassification,
            *,
            severities: Sequence[int],
            **kwargs) -> Tuple[T, T, T, T]:

        attack_permutation: Optional[torch.tensor] = kwargs.pop('attack_permutation', None)
        adv, success, distance, attack = self.attack(
            model, inputs, criterion, severities=severities, attack_permutation=attack_permutation, **kwargs
        )

        return adv, success, distance, attack

    def repeat(self, times: int) -> "RepeatedRotateImageCorruptionsAttack":
        return RepeatedRotateImageCorruptionsAttack(self, self.times * times)


def get_corruption_attacks(subset: Union[Tuple[str], str] = ('common',)) -> List[ImageCorruptionAttack]:
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
