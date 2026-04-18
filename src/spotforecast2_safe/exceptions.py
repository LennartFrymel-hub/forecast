# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

"""Custom exceptions and warnings for spotforecast2.

This module contains all the custom warnings and error classes used
across spotforecast2.

Examples:
    Using custom warnings::

        import warnings
        from spotforecast2_safe.exceptions import MissingValuesWarning

        # Raise a warning
        warnings.warn(
            "Missing values detected in input data.",
            MissingValuesWarning
        )

        # Suppress a specific warning
        warnings.simplefilter('ignore', category=MissingValuesWarning)
"""

import inspect
import textwrap
import warnings
from functools import wraps

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def runtime_deprecated(
    replacement: str = None,
    version: str = None,
    removal: str = None,
    category: type[Warning] = FutureWarning,
) -> object:
    """Decorator to mark functions or classes as deprecated.

    Works for both function and class targets, and ensures warnings are visible
    even inside Jupyter notebooks.

    Args:
        replacement: Name of the replacement function/class to use instead.
        version: Version in which the function/class was deprecated.
        removal: Version in which the function/class will be removed.
        category: Warning category to use. Default is FutureWarning.

    Returns:
        Decorator function.

    Examples:
        >>> @runtime_deprecated(replacement='new_function', version='0.5', removal='1.0')
        ... def old_function():
        ...     pass
        >>> old_function()  # doctest: +SKIP
        FutureWarning: old_function() is deprecated since version 0.5; use new_function instead...
    """

    def decorator(obj):
        is_function = inspect.isfunction(obj) or inspect.ismethod(obj)
        is_class = inspect.isclass(obj)

        if not (is_function or is_class):
            raise TypeError(
                "@runtime_deprecated can only be used on functions or classes"
            )

        # ----- Build warning message -----
        name = obj.__name__
        message = (
            f"{name}() is deprecated" if is_function else f"{name} class is deprecated"
        )
        if version:
            message += f" since version {version}"
        if replacement:
            message += f"; use {replacement} instead"
        if removal:
            message += f". It will be removed in version {removal}."
        else:
            message += "."

        def issue_warning():
            """Emit warning in a way that always shows in notebooks."""
            with warnings.catch_warnings():
                warnings.simplefilter("always", category)
                warnings.warn(message, category, stacklevel=3)

        # ----- Case 1: decorating a function -----
        if is_function:

            @wraps(obj)
            def wrapper(*args, **kwargs):
                issue_warning()
                return obj(*args, **kwargs)

            # Add metadata
            wrapper.__deprecated__ = True
            wrapper.__replacement__ = replacement
            wrapper.__version__ = version
            wrapper.__removal__ = removal
            return wrapper

        # ----- Case 2: decorating a class -----
        else:  # is_class must be True due to earlier check
            orig_init = getattr(obj, "__init__", None)
            orig_new = getattr(obj, "__new__", None)

            # Only wrap whichever exists (some classes use __new__, others __init__)
            if orig_new and (orig_new is not object.__new__):

                @wraps(orig_new)
                def wrapped_new(cls, *args, **kwargs):
                    issue_warning()
                    return orig_new(cls, *args, **kwargs)

                obj.__new__ = staticmethod(wrapped_new)

            elif orig_init:

                @wraps(orig_init)
                def wrapped_init(self, *args, **kwargs):
                    issue_warning()
                    return orig_init(self, *args, **kwargs)

                obj.__init__ = wrapped_init

            # Add metadata
            obj.__deprecated__ = True
            obj.__replacement__ = replacement
            obj.__version__ = version
            obj.__removal__ = removal

            return obj

    return decorator


class DataTypeWarning(UserWarning):
    """Warning for incompatible data types in exogenous data.

    Used to notify there are dtypes in the exogenous data that are not
    'int', 'float', 'bool' or 'category'. Most machine learning models do not
    accept other data types, therefore the forecaster `fit` and `predict` may fail.

    Examples:
        >>> import warnings
        >>> warnings.warn(
        ...     "Exogenous data contains unsupported dtypes.",
        ...     DataTypeWarning
        ... )  # doctest: +SKIP
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=DataTypeWarning)"
        )
        return self.message + "\\n" + extra_message


class DataTransformationWarning(UserWarning):
    """Warning for output data in transformed space.

    Used to notify that the output data is in the transformed space.

    Examples:
        >>> import warnings
        >>> warnings.warn(
        ...     "Output is in transformed space.",
        ...     DataTransformationWarning
        ... )  # doctest: +SKIP
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=DataTransformationWarning)"
        )
        return self.message + "\\n" + extra_message


class ExogenousInterpretationWarning(UserWarning):
    """Warning about implications when using exogenous variables.

    Used to notify about important implications when using exogenous
    variables with models that use a two-step approach (e.g., regression + ARAR).

    Examples:
        >>> import warnings
        >>> warnings.warn(
        ...     "Exogenous variables may not be used as expected.",
        ...     ExogenousInterpretationWarning
        ... )  # doctest: +SKIP
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=ExogenousInterpretationWarning)"
        )
        return self.message + "\\n" + extra_message


class FeatureOutOfRangeWarning(UserWarning):
    """Warning for features out of training range.

    Used to notify that a feature is out of the range seen during training.

    Examples:
        >>> import warnings
        >>> warnings.warn(
        ...     "Feature value exceeds training range.",
        ...     FeatureOutOfRangeWarning
        ... )  # doctest: +SKIP
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=FeatureOutOfRangeWarning)"
        )
        return self.message + "\\n" + extra_message


class IgnoredArgumentWarning(UserWarning):
    """Warning for ignored arguments.

    Used to notify that an argument is ignored when using a method
    or a function.

    Examples:
        >>> import warnings
        >>> warnings.warn(
        ...     "Argument 'x' is ignored in this context.",
        ...     IgnoredArgumentWarning
        ... )  # doctest: +SKIP
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=IgnoredArgumentWarning)"
        )
        return self.message + "\\n" + extra_message


class InputTypeWarning(UserWarning):
    """Warning for inefficient input format.

    Used to notify that input format is not the most efficient or
    recommended for the forecaster.

    Examples:
        >>> import warnings
        >>> warnings.warn(
        ...     "Input format is not optimal for this forecaster.",
        ...     InputTypeWarning
        ... )  # doctest: +SKIP
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=InputTypeWarning)"
        )
        return self.message + "\\n" + extra_message


class LongTrainingWarning(UserWarning):
    """Warning for potentially long training processes.

    Used to notify that a large number of models will be trained and the
    the process may take a while to run.

    Examples:
        >>> import warnings
        >>> warnings.warn(
        ...     "Training may take a long time.",
        ...     LongTrainingWarning
        ... )  # doctest: +SKIP
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=LongTrainingWarning)"
        )
        return self.message + "\\n" + extra_message


class MissingExogWarning(UserWarning):
    """Warning for missing exogenous variables.

    Used to indicate that there are missing exogenous variables in the
    data. Most machine learning models do not accept missing values, so the
    Forecaster's `fit' and `predict' methods may fail.

    Examples:
        >>> import warnings
        >>> warnings.warn(
        ...     "Missing exogenous variables detected.",
        ...     MissingExogWarning
        ... )  # doctest: +SKIP
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=MissingExogWarning)"
        )
        return self.message + "\\n" + extra_message


class MissingValuesWarning(UserWarning):
    """Warning for missing values in data.

    Used to indicate that there are missing values in the data. This
    warning occurs when the input data contains missing values, or the training
    matrix generates missing values. Most machine learning models do not accept
    missing values, so the Forecaster's `fit' and `predict' methods may fail.

    Args:
        message (str): The message to display.

    Examples:
        >>> import warnings
        >>> from spotforecast2_safe.exceptions import MissingValuesWarning
        >>> warnings.warn(
        ...     "Missing values detected in input data.",
        ...     MissingValuesWarning
        ... )  # doctest: +SKIP
    """

    def __init__(self, message: str) -> None:
        self.message = message

    def __str__(self) -> str:
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=MissingValuesWarning)"
        )
        return self.message + "\\n" + extra_message


class OneStepAheadValidationWarning(UserWarning):
    """Warning for one-step-ahead validation usage.

    Used to notify that the one-step-ahead validation is being used.

    Examples:
        >>> import warnings
        >>> warnings.warn(
        ...     "Using one-step-ahead validation.",
        ...     OneStepAheadValidationWarning
        ... )  # doctest: +SKIP
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=OneStepAheadValidationWarning)"
        )
        return self.message + "\\n" + extra_message


class ResidualsUsageWarning(UserWarning):
    """Warning for incorrect residuals usage.

    Used to notify that a residual are not correctly used in the
    probabilistic forecasting process.

    Examples:
        >>> import warnings
        >>> warnings.warn(
        ...     "Residuals are not properly used.",
        ...     ResidualsUsageWarning
        ... )  # doctest: +SKIP
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=ResidualsUsageWarning)"
        )
        return self.message + "\\n" + extra_message


class UnknownLevelWarning(UserWarning):
    """Warning for unknown levels in prediction.

    Used to notify that a level being predicted was not part of the
    training data.

    Examples:
        >>> import warnings
        >>> warnings.warn(
        ...     "Predicting for an unknown level.",
        ...     UnknownLevelWarning
        ... )  # doctest: +SKIP
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=UnknownLevelWarning)"
        )
        return self.message + "\\n" + extra_message


class SaveLoadSkforecastWarning(UserWarning):
    """Warning for save/load operations.

    Used to notify any issues that may arise when saving or loading
    a forecaster.

    Examples:
        >>> import warnings
        >>> warnings.warn(
        ...     "Issues detected when saving forecaster.",
        ...     SaveLoadSkforecastWarning
        ... )  # doctest: +SKIP
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=SaveLoadSkforecastWarning)"
        )
        return self.message + "\\n" + extra_message


class SpotforecastVersionWarning(UserWarning):
    """Warning for version mismatch.

    Used to notify that the version installed in the
    environment differs from the version used to initialize the forecaster.

    Examples:
        >>> import warnings
        >>> warnings.warn(
        ...     "Version mismatch detected.",
        ...     SpotforecastVersionWarning
        ... )  # doctest: +SKIP
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=SpotforecastVersionWarning)"
        )
        return self.message + "\\n" + extra_message


class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if estimator is used before fitting.

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.

    Examples:
        >>> from spotforecast2_safe.exceptions import NotFittedError
        >>> try:
        ...     raise NotFittedError("Forecaster not fitted")
        ... except NotFittedError as e:
        ...     print(e)
        Forecaster not fitted
    """


warn_skforecast_categories = [
    DataTypeWarning,
    DataTransformationWarning,
    ExogenousInterpretationWarning,
    FeatureOutOfRangeWarning,
    IgnoredArgumentWarning,
    InputTypeWarning,
    LongTrainingWarning,
    MissingExogWarning,
    MissingValuesWarning,
    OneStepAheadValidationWarning,
    ResidualsUsageWarning,
    UnknownLevelWarning,
    SaveLoadSkforecastWarning,
    SpotforecastVersionWarning,
]


def format_warning_handler(
    message: str,
    category: str,
    filename: str,
    lineno: str,
    file: object = None,
    line: str = None,
) -> None:
    """Custom warning handler to format warnings in a box.

    Args:
        message: Warning message.
        category: Warning category.
        filename: Filename where the warning was raised.
        lineno: Line number where the warning was raised.
        file: File where the warning was raised.
        line: Line where the warning was raised.

    Returns:
        None

    Examples:
        >>> # This is used internally by the warnings module
        >>> set_warnings_style('skforecast')  # doctest: +SKIP
    """

    if isinstance(message, tuple(warn_skforecast_categories)):
        width = 88
        title = type(message).__name__
        output_text = ["\\n"]

        wrapped_message = textwrap.fill(
            str(message), width=width - 2, expand_tabs=True, replace_whitespace=True
        )
        title_top_border = f"╭{'─' * ((width - len(title) - 2) // 2)} {title} {'─' * ((width - len(title) - 2) // 2)}╮"
        if len(title) % 2 != 0:
            title_top_border = title_top_border[:-1] + "─" + "╮"
        bottom_border = f"╰{'─' * width}╯"
        output_text.append(title_top_border)

        for line in wrapped_message.split("\\n"):
            output_text.append(f"│ {line.ljust(width - 2)} │")

        output_text.append(bottom_border)
        output_text = "\\n".join(output_text)
        color = "\\033[38;5;208m"
        reset = "\\033[0m"
        output_text = f"{color}{output_text}{reset}"
        print(output_text)
    else:
        # Fallback to default Python warning formatting
        warnings._original_showwarning(message, category, filename, lineno, file, line)


def rich_warning_handler(
    message: str,
    category: str,
    filename: str,
    lineno: str,
    file: object = None,
    line: str = None,
) -> None:
    """Custom handler for warnings that uses rich to display formatted panels.

    Args:
        message: Warning message.
        category: Warning category.
        filename: Filename where the warning was raised.
        lineno: Line number where the warning was raised.
        file: File where the warning was raised.
        line: Line where the warning was raised.

    Returns:
        None

    Examples:
        >>> # This is used internally when rich is available
        >>> set_warnings_style('skforecast')  # doctest: +SKIP
    """

    if isinstance(message, tuple(warn_skforecast_categories)):
        if not HAS_RICH:
            # Fallback to format_warning_handler if rich is not available
            format_warning_handler(message, category, filename, lineno, file, line)
            return

        console = Console()

        category_name = category.__name__
        text = (
            f"{message.message}\\n\\n"
            f"Category : spotforecast2.exceptions.{category_name}\\n"
            f"Location : {filename}:{lineno}\\n"
            f"Suppress : warnings.simplefilter('ignore', category={category_name})"
        )

        panel = Panel(
            Text(text, justify="left"),
            title=category_name,
            title_align="center",
            border_style="color(214)",
            width=88,
        )

        console.print(panel)
    else:
        # Fallback to default Python warning formatting
        warnings._original_showwarning(message, category, filename, lineno, file, line)


def set_warnings_style(style: str = "skforecast") -> None:
    """Set the warning handler based on the provided style.

    Args:
        style: The style of the warning handler. Either 'skforecast' or 'default'.

    Returns:
        None

    Examples:
        >>> set_warnings_style('skforecast')
        >>> # Now warnings will be displayed with formatting
        >>> set_warnings_style('default')
        >>> # Back to default Python warning format
    """
    if style == "skforecast":
        if not hasattr(warnings, "_original_showwarning"):
            warnings._original_showwarning = warnings.showwarning
        if HAS_RICH:
            warnings.showwarning = rich_warning_handler
        else:
            warnings.showwarning = format_warning_handler
    else:
        if hasattr(warnings, "_original_showwarning"):
            warnings.showwarning = warnings._original_showwarning


set_warnings_style(style="skforecast")


def set_skforecast_warnings(suppress_warnings: bool, action: str = "ignore") -> None:
    """
    Suppress spotforecast warnings.

    Args:
        suppress_warnings: bool
            If True, spotforecast warnings will be suppressed.
        action: str, default 'ignore'
            Action to take regarding the warnings.
    """
    if suppress_warnings:
        for category in warn_skforecast_categories:
            warnings.simplefilter(action, category=category)
