import logging
import sys


def setup_loggers():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)-8s] :: %(asctime)s :: %(name)-20s :: %(module)-15s :: %(funcName)-20s :: %(lineno)-4d :: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
