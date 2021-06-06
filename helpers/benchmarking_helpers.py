from functools import wraps
from DiplaCoreSDK.Configurations.spark import initialize_spark
import datetime
import logging
import inspect
import sys

# Need to improve this further, WIP method to improve spark UI logging.
# Can currently be a bit overly precise, so disabled for now except when needed.
# sys.settrace(lambda frame, e, arg: initialize_spark().sparkContext.setLocalProperty("callSite.short", f"crosssellpoc: {inspect.getframeinfo(frame)}"))

def apply_spark_job_logging_modifier(f):
    """
    Decorator function that is used to modify the spark callSite so that spark UI logs are more informative

    :param f: A function on a class that will have its qualified name added to the spark UI logs.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            # Update spark local properties so that the function name appears in the spark UI
            spark = initialize_spark(set_config=False)
            spark.sparkContext.setLocalProperty("callSite.short", f"crosssellpoc: {f.__qualname__}")

            # Call the function
            return f(*args, **kwargs)
        finally:
            pass
    return wrapper


def apply_benchmarking_metric_log(f):
    """
    Decorator function that is used to keep track of function start and end times in mlflow

    :param f: A function on a class that will have its qualified name added to the spark UI logs.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):

        start_time = datetime.datetime.now()

        # Call the function
        return_value = f(*args, **kwargs)

        end_time = datetime.datetime.now()

        logging_row = {
            "name": "benchmark_runtime",
            "function_name": f.__qualname__,
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": (end_time - start_time).total_seconds()
        }
        logging.info(f"Completed call to {f.__qualname__}", extra={"metrics_row": logging_row})

        return return_value
    return wrapper


def convert_time_delta_to_minutes(timedelta):
    return timedelta.days * 24 * 60 + timedelta.seconds / 60 + timedelta.microseconds

def for_all_methods(decorator):
    """
    Apply a decorator to all methods in the class.

    :param decorator: Takes in a decorator functoin which will be applied to all methods in the class
    """
    def decorate(cls):
        for attr in cls.__dict__:
            if callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls

    return decorate
