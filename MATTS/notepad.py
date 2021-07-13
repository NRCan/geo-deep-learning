import sys
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Sized
from dataclasses import dataclass, field
from datetime import timedelta
from math import ceil
from threading import Event, RLock, Thread
from types import TracebackType
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Iterable,
    List,
    NamedTuple,
    NewType,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from rich import filesize, get_console
from rich.console import Console, JustifyMethod, RenderableType, RenderGroup
from rich.highlighter import Highlighter
from rich.jupyter import JupyterMixin
from rich.live import Live
from rich.progress_bar import ProgressBar
from rich.spinner import Spinner
from rich.style import StyleType
from rich.table import  Table
from rich.text import Text, TextType


from time import sleep

from rich import inspect
from rich.panel import Panel
from rich.progress import FileSizeColumn, Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn
from rich.progress import Progress, Task
from rich.table import Table


class Progress_Pane(Progress):
    epoch_size = 0
    batch_size = 0
    epoch_curr = 0
    batch_curr = 0

    table = None

    def get_renderables(self):
        layout = Table.grid(expand=True)
        layout.add_column(justify="center")

        if len(self.tasks) == 1:
            # layout.add_row(Text('time:     since est.     remain     ', justify='right'))
            layout.add_row(Text('epoch:  '+str(self.epoch_curr+1) + ' / ' + str(self.epoch_size), style='purple', justify='left'))
            layout.add_row(self.make_tasks_table([self.tasks[0]]))
        elif len(self.tasks) > 1:
            # layout.add_row(Text('time: since est. remain', justify='right'))
            layout.add_row(Text('epoch:  '+str(self.epoch_curr+1) + ' / ' + str(self.epoch_size), style='purple', justify='left'))
            layout.add_row(self.make_tasks_table([self.tasks[0]]))
            layout.add_row(Text('   batch:   '+str(self.batch_curr+1) + ' / ' + str(self.batch_size), style='cyan1', justify='left'))
            layout.add_row(self.make_tasks_table(self.tasks[1:]))

        if self.table == None:
            yield Panel(layout, title='stats')
        else:
            bottom = Panel(layout, title='stats')
            lay = Table.grid(expand=True)
            lay.add_row(self.table)
            lay.add_row(bottom)
            yield Panel(lay)


    # def update_table(self):


if __name__ == '__main__':
    table = Table(title="[bold]\nThe Worst Star Wars[/bold] Movies", show_lines=True)
    table.add_column("Released", justify="center", style="cyan", no_wrap=True)

    table.add_row("Dec 20, 2019", "Star Wars: The Rise of Skywalker")
    table.add_row("May 25, 2018", "Solo: A Star Wars Story", "$393,151,347")

    with Progress_Pane(SpinnerColumn(                                              table_column=Column(ratio=1)),
                       # TextColumn("{task.description}",                            table_column=Column(ratio=3)),
                       TextColumn("[progress.percentage]{task.percentage:>3.0f}%", table_column=Column(ratio=1)),
                       BarColumn(                                                  table_column=Column(ratio=12)),
                       TimeElapsedColumn(                                          table_column=Column(ratio=2)),
                       TimeRemainingColumn(                                        table_column=Column(ratio=2)),
                       # console=console,
                       expand=True) as prgrss_pane:
        # prgrss_pane.setup()
        epoch_bar = prgrss_pane.add_task('[purple]epoch')
        batch_bar = prgrss_pane.add_task('[cyan1]batch')

        prgrss_pane.epoch_size = 3
        for n in prgrss_pane.track(range(6), description='epoch', task_id=0):
            prgrss_pane.epoch_curr = n

            prgrss_pane.reset(task_id=1)
            prgrss_pane.batch_size = n
            for nn in prgrss_pane.track(range(n), description='batch', task_id=1):
                prgrss_pane.batch_curr = nn
                prgrss_pane.table = table
                sleep(1)
            sleep(0.5)