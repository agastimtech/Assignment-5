import os

import aiohttp_jinja2
import jinja2
from aiohttp import web



def create_app():
    app = web.Application(client_max_size=1024 * 1024 * 5)
    jinja2_env = aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader('app/templates/'))
    return app