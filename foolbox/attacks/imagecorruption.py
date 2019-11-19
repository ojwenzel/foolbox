from functools import wraps
from typing import List
from warnings import warn

import numpy as np

from .base import Attack
from .base import generator_decorator
from ..criteria import Misclassification
from ..distances import MSE
from .. import nprng

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
def validate_corruptions(corruptions: List[str]) -> List[str]:
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


def get_subset(subset: List[str] = ['common']) -> List[str]:
    if 'common' in subset:
        return CORRUPTIONS[:15]
    elif 'validation' in subset:
        return CORRUPTIONS[15:]
    elif 'all' in subset or subset is None:
        return CORRUPTIONS
    else:
        return validate_corruptions(subset)


def list_corruption_attack_names(subset: List[str] = ['common']):
    """

    Retrieves class names for specified subset of corruption attacks. Note: identifiers 'common', 'all', 'validation'
    take precedence over individual corruption_names

    :param subset: specification of subset to retrieve may contain CORRUPTIONS + ['common', 'all', 'validation']
    :return:
    """
    corruptions = get_subset(subset)
    return [corruption_name_2_cls_name(corruption) for corruption in corruptions]


def get_corruption_attacks(subset: List[str] = ['common']):
    """

    Retrieves classes for specified subset of corruption attacks. Note: identifiers 'common', 'all', 'validation' take
    precedence over individual corruption_names

    :param subset: specification of subset to retrieve may contain CORRUPTIONS + ['common', 'all', 'validation']
    :return:
    """

    corruptions = list_corruption_attack_names(subset)
    return [globals()[corruption] for corruption in corruptions]


def image_transpose_decorator(func):
    """

    transposes images from format [channels, width , height] to [width, height, channels] before calling function

    :param func: image manipulation function
    :return: function wrapper that pre- and post-processes func input
    """
    @wraps(func)
    def with_transpose(image: np.ndarray, *args, **kwargs):
        image = image.transpose(1, 2, 0)
        image = func(image, *args, **kwargs)
        image = image.transpose(2, 0, 1)

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


@image_value_range_decorator
@image_transpose_decorator
def corrupt(image: np.ndarray, corruption: str, severity: int):
    return imagecorruptions.corrupt(image, corruption_name=corruption, severity=severity)


def to_255_image(image: np.ndarray) -> np.ndarray:
    return (image * 255).astype('uint8')


def to_0_1_image(image: np.ndarray) -> np.ndarray:
    return (image / 255).astype('float32')


# CLASSES
class GaussianNoise(Attack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'gaussian_noise'

    @generator_decorator
    def as_generator(self, a, severities: List[int] = None, repetitions=10, seed: int = None):

        """Increases the amount of specified image corruption until the input is misclassified.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        severities : List[int]
            severity-strengths of attack
        repetitions : int
            Specifies how often the attack will be repeated.
        seed : int
            sets a seed for reproducibility
        """
        # print('seed: {}'.format(seed))

        x = a.unperturbed
        severities = validate_severities(severities)

        for _ in range(repetitions):
            for s, severity in enumerate(severities):
                if seed is not None:
                    np.random.seed(seed)
                perturbed = corrupt(x, corruption=self.CORRUPTION, severity=severity)

                if a.normalized_distance(perturbed) >= a.distance:
                    continue

                _, is_adversarial = yield from a.forward_one(perturbed)
                if is_adversarial:
                    break

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class ShotNoise(Attack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'shot_noise'

    @generator_decorator
    def as_generator(self, a, severities: List[int] = None, repetitions=10, seed=None):

        """Increases the amount of specified image corruption until the input is misclassified.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        severities : List[int]
            severity-strengths of attack
        repetitions : int
            Specifies how often the attack will be repeated.
        seed : int
            sets a seed for reproducibility

        """

        x = a.unperturbed
        severities = validate_severities(severities)

        for _ in range(repetitions):
            for s, severity in enumerate(severities):
                if seed is not None:
                    np.random.seed(seed)
                perturbed = corrupt(x, corruption=self.CORRUPTION, severity=severity)

                if a.normalized_distance(perturbed) >= a.distance:
                    continue

                _, is_adversarial = yield from a.forward_one(perturbed)
                if is_adversarial:
                    break

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class ImpulseNoise(Attack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'impulse_noise'

    @generator_decorator
    def as_generator(self, a, severities: List[int] = None, repetitions=10, seed=None):

        """Increases the amount of specified image corruption until the input is misclassified.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        severities : List[int]
            severity-strengths of attack
        repetitions : int
            Specifies how often the attack will be repeated.
        seed : int
            sets a seed for reproducibility

        """

        x = a.unperturbed
        severities = validate_severities(severities)

        for _ in range(repetitions):
            for s, severity in enumerate(severities):
                if seed is not None:
                    np.random.seed(seed)
                perturbed = corrupt(x, corruption=self.CORRUPTION, severity=severity)

                if a.normalized_distance(perturbed) >= a.distance:
                    continue

                _, is_adversarial = yield from a.forward_one(perturbed)
                if is_adversarial:
                    break

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class DefocusBlur(Attack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'defocus_blur'

    @generator_decorator
    def as_generator(self, a, severities: List[int] = None, repetitions=10, seed=None):

        """Increases the amount of specified image corruption until the input is misclassified.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        severities : List[int]
            severity-strengths of attack
        repetitions : int
            Specifies how often the attack will be repeated.
        seed : int
            sets a seed for reproducibility

        """

        x = a.unperturbed
        severities = validate_severities(severities)

        for _ in range(repetitions):
            for s, severity in enumerate(severities):
                if seed is not None:
                    np.random.seed(seed)
                perturbed = corrupt(x, corruption=self.CORRUPTION, severity=severity)

                if a.normalized_distance(perturbed) >= a.distance:
                    continue

                _, is_adversarial = yield from a.forward_one(perturbed)
                if is_adversarial:
                    break

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class GlassBlur(Attack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'glass_blur'

    @generator_decorator
    def as_generator(self, a, severities: List[int] = None, repetitions=10, seed=None):

        """Increases the amount of specified image corruption until the input is misclassified.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        severities : List[int]
            severity-strengths of attack
        repetitions : int
            Specifies how often the attack will be repeated.
        seed : int
            sets a seed for reproducibility

        """

        x = a.unperturbed
        severities = validate_severities(severities)

        for _ in range(repetitions):
            for s, severity in enumerate(severities):
                if seed is not None:
                    np.random.seed(seed)
                perturbed = corrupt(x, corruption=self.CORRUPTION, severity=severity)

                if a.normalized_distance(perturbed) >= a.distance:
                    continue

                _, is_adversarial = yield from a.forward_one(perturbed)
                if is_adversarial:
                    break

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class MotionBlur(Attack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'motion_blur'

    @generator_decorator
    def as_generator(self, a, severities: List[int] = None, repetitions=10, seed=None):

        """Increases the amount of specified image corruption until the input is misclassified.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        severities : List[int]
            severity-strengths of attack
        repetitions : int
            Specifies how often the attack will be repeated.
        seed : int
            sets a seed for reproducibility

        """

        x = a.unperturbed
        severities = validate_severities(severities)

        for _ in range(repetitions):
            for s, severity in enumerate(severities):
                if seed is not None:
                    np.random.seed(seed)
                perturbed = corrupt(x, corruption=self.CORRUPTION, severity=severity)

                if a.normalized_distance(perturbed) >= a.distance:
                    continue

                _, is_adversarial = yield from a.forward_one(perturbed)
                if is_adversarial:
                    break

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class ZoomBlur(Attack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'zoom_blur'

    @generator_decorator
    def as_generator(self, a, severities: List[int] = None, repetitions=10, seed=None):

        """Increases the amount of specified image corruption until the input is misclassified.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        severities : List[int]
            severity-strengths of attack
        repetitions : int
            Specifies how often the attack will be repeated.
        seed : int
            sets a seed for reproducibility

        """

        x = a.unperturbed
        severities = validate_severities(severities)

        for _ in range(repetitions):
            for s, severity in enumerate(severities):
                if seed is not None:
                    np.random.seed(seed)
                perturbed = corrupt(x, corruption=self.CORRUPTION, severity=severity)

                if a.normalized_distance(perturbed) >= a.distance:
                    continue

                _, is_adversarial = yield from a.forward_one(perturbed)
                if is_adversarial:
                    break

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class Snow(Attack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'snow'

    @generator_decorator
    def as_generator(self, a, severities: List[int] = None, repetitions=10, seed=None):

        """Increases the amount of specified image corruption until the input is misclassified.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        severities : List[int]
            severity-strengths of attack
        repetitions : int
            Specifies how often the attack will be repeated.
        seed : int
            sets a seed for reproducibility

        """

        x = a.unperturbed
        severities = validate_severities(severities)

        for _ in range(repetitions):
            for s, severity in enumerate(severities):
                if seed is not None:
                    np.random.seed(seed)
                perturbed = corrupt(x, corruption=self.CORRUPTION, severity=severity)

                if a.normalized_distance(perturbed) >= a.distance:
                    continue

                _, is_adversarial = yield from a.forward_one(perturbed)
                if is_adversarial:
                    break

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class Frost(Attack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'frost'

    @generator_decorator
    def as_generator(self, a, severities: List[int] = None, repetitions=10, seed=None):

        """Increases the amount of specified image corruption until the input is misclassified.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        severities : List[int]
            severity-strengths of attack
        repetitions : int
            Specifies how often the attack will be repeated.
        seed : int
            sets a seed for reproducibility

        """

        x = a.unperturbed
        severities = validate_severities(severities)

        for _ in range(repetitions):
            for s, severity in enumerate(severities):
                if seed is not None:
                    np.random.seed(seed)
                perturbed = corrupt(x, corruption=self.CORRUPTION, severity=severity)

                if a.normalized_distance(perturbed) >= a.distance:
                    continue

                _, is_adversarial = yield from a.forward_one(perturbed)
                if is_adversarial:
                    break

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class Fog(Attack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'fog'

    @generator_decorator
    def as_generator(self, a, severities: List[int] = None, repetitions=10, seed=None):

        """Increases the amount of specified image corruption until the input is misclassified.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        severities : List[int]
            severity-strengths of attack
        repetitions : int
            Specifies how often the attack will be repeated.
        seed : int
            sets a seed for reproducibility

        """

        x = a.unperturbed
        severities = validate_severities(severities)

        for _ in range(repetitions):
            for s, severity in enumerate(severities):
                if seed is not None:
                    np.random.seed(seed)
                perturbed = corrupt(x, corruption=self.CORRUPTION, severity=severity)

                if a.normalized_distance(perturbed) >= a.distance:
                    continue

                _, is_adversarial = yield from a.forward_one(perturbed)
                if is_adversarial:
                    break

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class Brightness(Attack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'brightness'

    @generator_decorator
    def as_generator(self, a, severities: List[int] = None, repetitions=10, seed=None):

        """Increases the amount of specified image corruption until the input is misclassified.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        severities : List[int]
            severity-strengths of attack
        repetitions : int
            Specifies how often the attack will be repeated.
        seed : int
            sets a seed for reproducibility

        """

        x = a.unperturbed
        severities = validate_severities(severities)

        for _ in range(repetitions):
            for s, severity in enumerate(severities):
                if seed is not None:
                    np.random.seed(seed)
                perturbed = corrupt(x, corruption=self.CORRUPTION, severity=severity)

                if a.normalized_distance(perturbed) >= a.distance:
                    continue

                _, is_adversarial = yield from a.forward_one(perturbed)
                if is_adversarial:
                    break

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class Contrast(Attack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'contrast'

    @generator_decorator
    def as_generator(self, a, severities: List[int] = None, repetitions=10, seed=None):

        """Increases the amount of specified image corruption until the input is misclassified.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        severities : List[int]
            severity-strengths of attack
        repetitions : int
            Specifies how often the attack will be repeated.
        seed : int
            sets a seed for reproducibility

        """

        x = a.unperturbed
        severities = validate_severities(severities)

        for _ in range(repetitions):
            for s, severity in enumerate(severities):
                if seed is not None:
                    np.random.seed(seed)
                perturbed = corrupt(x, corruption=self.CORRUPTION, severity=severity)

                if a.normalized_distance(perturbed) >= a.distance:
                    continue

                _, is_adversarial = yield from a.forward_one(perturbed)
                if is_adversarial:
                    break

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class ElasticTransform(Attack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'elastic_transform'

    @generator_decorator
    def as_generator(self, a, severities: List[int] = None, repetitions=10, seed=None):

        """Increases the amount of specified image corruption until the input is misclassified.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        severities : List[int]
            severity-strengths of attack
        repetitions : int
            Specifies how often the attack will be repeated.
        seed : int
            sets a seed for reproducibility

        """

        x = a.unperturbed
        severities = validate_severities(severities)

        for _ in range(repetitions):
            for s, severity in enumerate(severities):
                if seed is not None:
                    np.random.seed(seed)
                perturbed = corrupt(x, corruption=self.CORRUPTION, severity=severity)

                if a.normalized_distance(perturbed) >= a.distance:
                    continue

                _, is_adversarial = yield from a.forward_one(perturbed)
                if is_adversarial:
                    break

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class Pixelate(Attack):
    """Increases the amount of image corruption until the input is misclassified.

    """

    CORRUPTION = 'pixelate'

    @generator_decorator
    def as_generator(self, a, severities: List[int] = None, repetitions=10, seed=None):

        """Increases the amount of specified image corruption until the input is misclassified.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        severities : List[int]
            severity-strengths of attack
        repetitions : int
            Specifies how often the attack will be repeated.
        seed : int
            sets a seed for reproducibility

        """

        x = a.unperturbed
        severities = validate_severities(severities)

        for _ in range(repetitions):
            for s, severity in enumerate(severities):
                if seed is not None:
                    np.random.seed(seed)
                perturbed = corrupt(x, corruption=self.CORRUPTION, severity=severity)

                if a.normalized_distance(perturbed) >= a.distance:
                    continue

                _, is_adversarial = yield from a.forward_one(perturbed)
                if is_adversarial:
                    break

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)


class JpegCompression(Attack):
    """Increases the amount of image corruption until the input is misclassified.

    """
    
    CORRUPTION = 'jpeg_compression'

    @generator_decorator
    def as_generator(self, a, severities: List[int] = None, repetitions=10, seed=None):

        """Increases the amount of specified image corruption until the input is misclassified.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        severities : List[int]
            severity-strengths of attack
        repetitions : int
            Specifies how often the attack will be repeated.
        seed : int
            sets a seed for reproducibility

        """

        x = a.unperturbed
        severities = validate_severities(severities)

        for _ in range(repetitions):
            for s, severity in enumerate(severities):
                if seed is not None:
                    np.random.seed(seed)
                perturbed = corrupt(x, corruption=self.CORRUPTION, severity=severity)

                if a.normalized_distance(perturbed) >= a.distance:
                    continue

                _, is_adversarial = yield from a.forward_one(perturbed)
                if is_adversarial:
                    break

    def __str__(self):
        return 'Corruption attack of type \'{}\''.format(self.CORRUPTION)
