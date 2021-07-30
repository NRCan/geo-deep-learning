from rich.table import Table, Column
from rich.progress import _TrackThread, Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn
from rich.panel import Panel
from rich.style import StyleType
from rich.text import Text, TextType
from typing import Iterable, Optional, Sequence, TypeVar, Union, NewType
from collections.abc import Sized
from rich.console import Console

TaskID = NewType("TaskID", int)
ProgressType = TypeVar("ProgressType")


class Tracking_Pane(Progress):

    info = ''

    def __init__(self, console=None, mode=None, other_renderables=[]):
        self.other_renderables = other_renderables

        self.mode = mode
        if mode == 'train':
            super().__init__(
                             SpinnerColumn(table_column=Column(ratio=1)),
                             TextColumn("{task.description}", table_column=Column(ratio=3)),
                             # TextColumn("[progress.percentage]{task.percentage:>3.0f}%", table_column=Column(ratio=1)),
                             TextColumn("[progress.percentage]{task.completed} / {task.total:>.0f}", style='bold white', table_column=Column(ratio=3)),
                             BarColumn(table_column=Column(ratio=10)),
                             TimeElapsedColumn(table_column=Column(ratio=2)),
                             TimeRemainingColumn(table_column=Column(ratio=2)),
                             auto_refresh=False,
                             console=console,
                             expand=True)
            self.add_task('[cyan1]epoch')
            self.add_task('[bright_magenta]trn batch')
            self.add_task('[magenta]val batch')
        elif mode == 'im_to_samp':
            super().__init__(
                         SpinnerColumn(table_column=Column(ratio=1)),
                         TextColumn("{task.description}", table_column=Column(ratio=3)),
                         # TextColumn("[progress.percentage]{task.percentage:>3.0f}%", table_column=Column(ratio=1)),
                         TextColumn("[progress.percentage]{task.completed} / {task.total:>.0f}", style='bold white', table_column=Column(ratio=3)),
                         BarColumn(table_column=Column(ratio=10)),
                         TimeElapsedColumn(table_column=Column(ratio=2)),
                         auto_refresh=False,
                         console=console,
                         expand=True)
            self.add_task('[cyan1]csv row')
            self.add_task('[bright_magenta]im row')
            self.add_task('[magenta]im column')
            self.trn_samples = 0
            self.idx_samples_v = 0
            self.tst_samples = 0
            self.excl_samples = 0
            self.num_padded = 0



    def get_renderables(self):
        layout = Table.grid(expand=True)
        layout.add_column(justify="center")

        # if len(self.tasks) == 1:
        #     # layout.add_row(Text('time:     since est.     remain     ', justify='right'))
        #     # layout.add_row(Text('epoch:  '+str(self.epoch_curr+1) + ' / ' + str(self.epoch_size), style='purple', justify='left'))
        #     layout.add_row(self.make_tasks_table([self.tasks[0]]))
        if len(self.tasks) > 1:
            # layout.add_row(Text('time: since est. remain', justify='right'))
            # layout.add_row(Text('epoch:  '+str(self.epoch_curr+1) + ' / ' + str(self.epoch_size), style='purple', justify='left'))
            # layout.add_row(Text('   batch:   '+str(self.batch_curr+1) + ' / ' + str(self.batch_size), style='cyan1', justify='left'))
            layout.add_row(self.make_tasks_table(self.tasks))
            # layout.add_row(self.make_tasks_table([self.tasks[0]]))
            # layout.add_row(self.make_tasks_table(self.tasks[1:]))
            if not self.info == '':
                layout.add_row(self.info)
            if self.mode == 'im_to_samp':
                layout.add_row(f'trn:[green]{self.trn_samples}[/]   val:[orange1]{self.idx_samples_v}[/]   tst:[blue]{self.tst_samples}[/]   under_min%:[red]{self.excl_samples}[/]   padded:[yellow1]{self.num_padded}[/]')

        if not len(self.other_renderables) > 0:
            yield Panel(layout)
        else:
            layout_big = Table.grid(expand=True)
            for renderable in self.other_renderables:
                layout_big.add_row(renderable)
            layout_big.add_row(Panel(layout))
            yield Panel(layout_big)

    # def track(
    #     self,
    #     sequence: Union[Iterable[ProgressType], Sequence[ProgressType]],
    #     total: Optional[float] = None,
    #     task_id: Optional[TaskID] = None,
    #     description: str = "Working...",
    #     update_period: float = 0.1,
    # ) -> Iterable[ProgressType]:
    #     """Track progress by iterating over a sequence.
    #
    #     Args:
    #         sequence (Sequence[ProgressType]): A sequence of values you want to iterate over and track progress.
    #         total: (float, optional): Total number of steps. Default is len(sequence).
    #         task_id: (TaskID): Task to track. Default is new task.
    #         description: (str, optional): Description of task, if new task is created.
    #         update_period (float, optional): Minimum time (in seconds) between calls to update(). Defaults to 0.1.
    #
    #     Returns:
    #         Iterable[ProgressType]: An iterable of values taken from the provided sequence.
    #     """
    #
    #     if total is None:
    #         if isinstance(sequence, Sized):
    #             task_total = float(len(sequence))
    #         else:
    #             raise ValueError(
    #                 f"unable to get size of {sequence!r}, please specify 'total'"
    #             )
    #     else:
    #         task_total = total
    #
    #     if task_id is None:
    #         task_id = self.add_task(description, total=task_total)
    #     else:
    #         self.update(task_id, total=task_total)
    #
    #     if self.live.auto_refresh:
    #         with _TrackThread(self, task_id, update_period) as track_thread:
    #             for value in sequence:
    #                 yield value
    #                 track_thread.completed += 1
    #     else: # ignore, unless you disabled auto_refresh
    #         advance = self.advance
    #         refresh = self.refresh
    #         for value in sequence:
    #             yield value
    #             advance(task_id, 1)
    #             refresh()