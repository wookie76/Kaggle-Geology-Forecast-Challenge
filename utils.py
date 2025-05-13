"""
General utility functions for the Geology Forecast Challenge pipeline.
This module can house small, miscellaneous helper functions that don't
belong to more specific components like data_io, preprocessing, evaluation, etc.
"""

from loguru import logger


# Example utility (if any common, small, non-domain specific helpers arise)
def example_utility_function(param1: int, param2: str) -> bool:
    """
    An example utility function.
    Replace or remove this as actual utilities are identified.
    """
    logger.debug(f"Example utility called with {param1}, {param2}")
    return True


# If no general utilities are identified, this file can be minimal or removed
# and its imports adjusted. For now, keeping it as a placeholder.
