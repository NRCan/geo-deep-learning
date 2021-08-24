import time
import shutil
from datetime import datetime
from pathlib import Path
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

    def __init__(self, dir_path, mode=None, debug=False):
        self.start_time = {}

        if mode not in ['trn_seg', 'im_to_samp']:
            raise NotImplementedError(f'please select a valid mode (ur choices = ["trn_seg", "im_to_samp"])')

        # region create Directory
        self.dir = Path(dir_path) / 'logs'
        print(self.dir)
        if self.dir.is_dir():
            if debug: # TODO: do we need this if we already check for samples_dir before hand? ...yes? for if other than im_to_samp.pyit will not check, correct?
                # Move existing data folder with a random suffix.
                last_mod_time_suffix = datetime.fromtimestamp(self.dir.stat().st_mtime).strftime('%Y%m%d-%H%M%S')
                shutil.move(self.dir, self.dir.joinpath(f'{str(self.dir)}_{last_mod_time_suffix}'))
            else:
                raise FileExistsError(f'Data path exists: {self.dir}. Remove it or use a different experiment_name.')
        Path.mkdir(self.dir, exist_ok=False)
        # endregion

        if mode == 'trn_seg':
            for file_type in ['per_epoch.csv', 'per_batch.csv', 'live.txt']:
                with open(self.dir / file_type, 'a') as file:

                    if file_type == 'live.txt':
                        file.writelines(('               curr     total    |     time elapsed      est. time remain    est. total time\n',
                                         'epoch       :\n',
                                         'trn batch   :\n',
                                         'val batch   :\n',
                                         '\n'))

                    elif file_type == 'per_epoch.csv':
                        file.writelines(('epoch,\n',
                                         '\n'))
        # elif mode == 'im_to_samp':
        self.mode = mode


    def track(self, seq, task):
        # todo: add Not ImplementedERR for task
        for num, value in enumerate(seq):

            if self.mode == 'trn_seg':


                # region add to live.txt file
                with open(self.dir / 'live.txt', 'r') as file:
                    data = file.readlines()

                for d in range(len(data)):
                    if data[d][:len(task)] == task:

                        if num == 0:
                            self.start_time[task] = time.time()
                            time_curr = 0
                            time_total = time_curr * len(seq)
                        else:
                            time_curr = time.time() - self.start_time[task]
                            time_total = time_curr / num * (len(seq)-1)

                        time_info = f"{(time_curr / 60):.0f}m " \
                                    f"{(time_curr % 60):.0f}s"
                        time_remain = f"{((time_total - time_curr) / 60):.0f}m " \
                                      f"{((time_total - time_curr) % 60):.0f}s"

                        data[d] = f"{task}{' '*(12-len(task))}:   " \
                                  f"{num+1}{' '*(4-len(str(num+1)))} /   " \
                                  f"{len(seq)}{' '*(4-len(str(len(seq))))}           " \
                                  f"{time_info}{' '*(12-len(time_info))}         " \
                                  f"{time_remain}{' '*(12-len(time_remain))}        " \
                                  f"{(time_total / 60):.0f}m " \
                                  f"{(time_total % 60):.0f}s\n"

                        if task == 'epoch':
                            data[d+1] = "trn batch   :\n"
                            data[d+2] = "val batch   :\n"

                with open(self.dir / 'live.txt', 'w') as file:
                    file.writelines(data)                           # todo: except possible file writing errs!!!
                # endregion

            # self.print(task, ' :   ', value)
            yield value


if __name__ == '__main__':
    batches = [[0, 1, 2, 3, 55],
               [0, 1, 2, 3],
               [0, 1, 2, 3, 55, 33]]
    tracker = Tracking_Pane('C:/Users/muzwe/Documents/GitHub/geo-deep-learning - MATT/MATTS', mode='trn_seg', debug=True)
    for i in tracker.track(range(6), 'epoch'):#task_id=0):
        for j in tracker.track(batches[i % 3], 'trn batch'):#task_id=1):
            # print(len(batches[i % 4]), batches[i % 4])
            # tracker.print('\t\t\t',i, j)
            time.sleep(1)
            xxx=69