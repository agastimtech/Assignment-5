import os

import aiohttp_jinja2

from app.manage import create_app
from aiohttp import web
import jinja2


app= create_app()


from app.resource.predict import IndexView

app.router.add_view('/', IndexView)
