import numpy as np

try:
    import ashjdfioastkjl
    from rich.table import Table, Column
    from rich.progress import Task, _TrackThread, Progress, BarColumn, TextColumn, TimeElapsedColumn, \
        TimeRemainingColumn, SpinnerColumn
    from rich.panel import Panel
    from rich.style import StyleType
    from rich.text import Text, TextType
    from rich.console import Console

    RICH = True

except ModuleNotFoundError:
    print('rich not imported')

    RICH = False
    console_width = 78  # preferably an even number (but not required)


    class Progress():
        def __init__(self):
            print('*' * 80)
            print('rich not imported, using simpler version instead, console width =', console_width)

from typing import Iterable, Optional, Sequence, TypeVar, Union, NewType, Any
from collections.abc import Sized

TaskID = NewType("TaskID", int)
ProgressType = TypeVar("ProgressType")


class Tracking_Pane(Progress):
    """
    if using rich library

    if NOT
        will print updates each time track is called

        params:     mode = either('train' or 'im_to_samp')
    """

    info = ''

    def __init__(self, mode=None, other_renderables=[]):
        self.taskss = {}
        if RICH:
            # self.console = Console() # TODO can prob just use parent class's console instead

            self.other_renderables = other_renderables
            self.mode = mode
            if mode == 'train':
                super().__init__(SpinnerColumn(table_column=Column(ratio=1)),
                                 TextColumn("{task.description}", table_column=Column(ratio=3)),
                                 # TextColumn("[progress.percentage]{task.percentage:>3.0f}%", table_column=Column(ratio=1)),
                                 TextColumn("[progress.percentage]{task.completed} / {task.total:>.0f}",
                                            style='bold white', table_column=Column(ratio=3)),
                                 BarColumn(table_column=Column(ratio=10)),
                                 TimeElapsedColumn(table_column=Column(ratio=2)),
                                 TimeRemainingColumn(table_column=Column(ratio=2)),
                                 auto_refresh=False,
                                 console=None,
                                 expand=True)
                # self.taskss['epoch'] = self.add_task('[cyan1]epoch')
                # self.taskss['trn batch'] = self.add_task('[bright_magenta]trn batch')
                # self.taskss['val batch'] = self.add_task('[magenta]val batch')

            elif mode == 'im_to_samp':
                super().__init__(SpinnerColumn(table_column=Column(ratio=1)),
                                 TextColumn("{task.description}", table_column=Column(ratio=3)),
                                 # TextColumn("[progress.percentage]{task.percentage:>3.0f}%", table_column=Column(ratio=1)),
                                 TextColumn("[progress.percentage]{task.completed} / {task.total:>.0f}",
                                            style='bold white', table_column=Column(ratio=3)),
                                 BarColumn(table_column=Column(ratio=10)),
                                 TimeElapsedColumn(table_column=Column(ratio=2)),
                                 auto_refresh=False,
                                 console=None,
                                 expand=True)
                self.taskss['csv row'] = self.add_task('[cyan1]csv row')
                self.taskss['im row'] = self.add_task('[bright_magenta]im row')
                self.taskss['im column'] = self.add_task('[magenta]im column')
                self.trn_samples = 0
                self.idx_samples_v = 0
                self.tst_samples = 0
                self.excl_samples = 0
                self.num_padded = 0
        else:
            # self.other_renderables = other_renderables # TODO: add implementation err (or something)
            self.mode = mode
            # if mode == 'train':
            # elif mode == 'im_to_samp':

    # def add_task(
    #     self,
    #     description: str,
    #     start: bool = True,
    #     total: float = 100.0,
    #     completed: int = 0,
    #     visible: bool = True,
    #     **fields: Any,
    # ) -> TaskID:
    #     """Add a new 'task' to the Progress display.
    #
    #     Args:
    #         description (str): A description of the task.
    #         start (bool, optional): Start the task immediately (to calculate elapsed time). If set to False,
    #             you will need to call `start` manually. Defaults to True.
    #         total (float, optional): Number of total steps in the progress if know. Defaults to 100.
    #         completed (int, optional): Number of steps completed so far.. Defaults to 0.
    #         visible (bool, optional): Enable display of the task. Defaults to True.
    #         **fields (str): Additional data fields required for rendering.
    #
    #     Returns:
    #         TaskID: An ID you can use when calling `update`.
    #     """
    #     with self._lock:
    #         task = Task(
    #             self._task_index,
    #             description,
    #             total,
    #             completed,
    #             visible=visible,
    #             fields=fields,
    #             _get_time=self.get_time,
    #             _lock=self._lock,
    #         )
    #         self._tasks[self._task_index] = task
    #         if start:
    #             self.start_task(self._task_index)
    #         new_task_index = self._task_index
    #         self._task_index = TaskID(int(self._task_index) + 1)
    #     self.refresh()
    #     return new_task_index

    def get_renderables(self):
        if RICH:
            layout = Table.grid(expand=True)
            layout.add_column(justify="center")

            # if len(self.taskss) == 1:
            #     # layout.add_row(Text('time:     since est.     remain     ', justify='right'))
            #     # layout.add_row(Text('epoch:  '+str(self.epoch_curr+1) + ' / ' + str(self.epoch_size), style='purple', justify='left'))
            #     layout.add_row(self.make_tasks_table([self.tasks[0]]))
            if len(self.taskss) > 1:
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
        # else:

    def track(
            self,
            sequence: Union[Iterable[ProgressType], Sequence[ProgressType]],
            task_key,
            total: Optional[float] = None,
            # task_id: Optional[TaskID] = None,
            # description: str = "Working...",
            update_period: float = 0.1,
    ) -> Iterable[ProgressType]:
        """Track progress by iterating over a sequence.

        Args:
            sequence (Sequence[ProgressType]): A sequence of values you want to iterate over and track progress.
            total: (float, optional): Total number of steps. Default is len(sequence).
            task_id: (TaskID): Task to track. Default is new task.
            description: (str, optional): Description of task, if new task is created.
            update_period (float, optional): Minimum time (in seconds) between calls to update(). Defaults to 0.1.

        Returns:
            Iterable[ProgressType]: An iterable of values taken from the provided sequence.
        """
        if task_key not in self.taskss:
            self.taskss[task_key] = {'curr' : 0,
                                     'total': len(sequence),
                                     'sequence': sequence}
        else:
            self.taskss[task_key]['curr'] += 1
            self.taskss[task_key]['total'] = len(sequence)

        RESET_others = False
        for key in self.taskss:
            if RESET_others:
                self.taskss[key]['curr'] = 0
            if task_key == key:
                RESET_others = True

        if RICH:
            if self.mode == 'train':
                if task_key == 'epoch':
                    task_id=0
                elif task_key == 'trn batch':
                    task_id=1
                elif task_key == 'val batch':
                    task_id=2
                elif task_key == 'tst batch':
                    task_id=3

            elif self.mode == 'im_to_samp':
                if task_key == 'csv':
                    task_id=0 # TODO finish im->s task_ids


            # if task_key not in self.taskss:
            #     self.taskss[task_key] = {'curr' : 0,
            #                             'total': len(sequence)}
            # else:
            if total is None:
                if isinstance(sequence, Sized):
                    task_total = float(len(sequence))
                else:
                    raise ValueError(
                        f"unable to get size of {sequence!r}, please specify 'total'"
                    )
            else:
                task_total = total

            # if task_id is None:
            #     task_id = self.add_task(description, total=task_total)
            # else:
            #     self.update(task_id, total=task_total)

            if self.live.auto_refresh:
                with _TrackThread(self, task_id, update_period) as track_thread:
                    for value in sequence:
                        yield value
                        track_thread.completed += 1
            else:  # ignore, unless you disabled auto_refresh
                advance = self.advance
                refresh = self.refresh
                for value in sequence:
                    yield value
                    advance(task_id, 1)
                    refresh()
        else:
            if task_key not in self.taskss:
                self.taskss[task_key] = {'curr' : 0,
                                         'total': len(sequence)}

            for value in self.taskss[task_key]['sequence']:
                self.taskss[task_key]['curr'] += 1
                self.taskss[task_key]['total'] = len(sequence)

                s = []
                RESET_others = False

                for key in self.taskss:

                    if RESET_others:
                        self.taskss[key]['curr'] = 0
                    if task_key == key:
                        RESET_others = True

                        s.append(f'-->   {key} : {self.taskss[key]["curr"]} / {self.taskss[key]["total"]}      ')
                    else:
                        s.append(f'{key} : {self.taskss[key]["curr"]} / {self.taskss[key]["total"]}')
                self.print(s, style='thick panel')

                yield value

    def print(self, *objects: Any, style=None, justify=None):
        if RICH:
            self.console.print(*objects, style=style, justify=justify)
        else:
            if style == 'thick panel':
                print('#' * console_width)
                # print('###' + ' '*(console_width-6) + '###')
                for obj in objects[0]:
                    obj = str(obj)
                    buffer_size = (console_width - 6 - len(obj)) // 2
                    print('###' + ' ' * buffer_size + obj + ' ' * (buffer_size + len(obj) % 2) + '###')
                # print('###' + ' '*(console_width-6) + '###')
                print('#' * console_width)
            else:
                print(*objects)


if __name__ == '__main__':
    tracker = Tracking_Pane()
    for i in tracker.track([0, 1, 2, 3, 55], 'epoch'):
        for j in tracker.track([0, 1, 2, 3, 55], 'trn batch'):
            tracker.print('\t\t\t\t', i, j, [0, 1])
