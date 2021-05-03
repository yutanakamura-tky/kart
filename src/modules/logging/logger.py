import datetime
import logging
import pathlib
import sys


def get_datetime_datetime_now() -> datetime.datetime:
    """
    Get current year, month, & date in datetime.datetime
    """
    jst = datetime.timezone(datetime.timedelta(hours=9))
    return datetime.datetime.now(jst)


def get_time_struct_time_now(*args) -> datetime.datetime:
    """
    Get current year, month, date, & time in time.struct_time
    Passing this function to logging.Formatter().converter attribute will change logging time from GMT to JST.
    """
    return get_datetime_datetime_now().timetuple()


def get_stream_handler(level: int = logging.INFO) -> logging.Handler:
    plain_log_formatter = logging.Formatter("{message}", style="{")
    plain_log_formatter.converter = get_time_struct_time_now
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(plain_log_formatter)
    return stream_handler


def get_file_handler(
    log_save_path: pathlib.PosixPath, level: int = logging.DEBUG
) -> logging.Handler:
    datetime_log_formatter = logging.Formatter(
        "{asctime} - {levelname} - {filename} - {name} - {funcName} - " + "{message}",
        style="{",
    )
    datetime_log_formatter.converter = get_time_struct_time_now
    file_handler = logging.FileHandler(log_save_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(datetime_log_formatter)
    return file_handler
