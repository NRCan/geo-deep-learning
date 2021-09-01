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
    # TODO add a notepad.txt file & let ppl add anything to it
    '''
    WARNINGS:
        try and make your titles less than 12 chars long for better formatting :)
        csv stats must always be the same # TODO: add a check to make sure stats are the same
        each time .track is called it MUST use a new/diff task param NOT the same

        PARAMS:

        - dir_path: directory location to save files

        - mode: str = "trn_seg" or "im_to_samp"

        - stats_to_track:
                          dict{ task : [column_title,
                                        column_title,
                                        [column_title, custom_indent_size],     NOTE: default indent = 12
                                        column_title,
                                        etc...],
                                etc...}

    '''
    def __init__(self, dir_path, mode=None, stats_to_track={'epoch' : [], 'batch' : []}):

        # region INITS

        # general inits
        self.dir = dir_path
        if mode not in ['trn_seg', 'im_to_samp']:
            raise NotImplementedError(f'please select a valid mode (ur choices = ["trn_seg", "im_to_samp"])')
        self.mode = mode
        self.start_time = {}

        # init stat dictionaries
        self.stats = {}
        self.stat_cols = {}
        self.stat_indents = {}
        for task in stats_to_track:
            self.stats[task] = {}
            self.stat_cols[task] = []
            self.stat_indents[task] = []

            # add column_titles & default indent size
            self.stat_cols[task] += stats_to_track[task]
            self.stat_indents[task] += [12] * len(stats_to_track[task])

            # check for specified indent sizes
            for c in range(len(self.stat_cols[task])):
                if isinstance(self.stat_cols[task][c], list):
                    self.stat_indents[task][c] = self.stat_cols[task][c][1]
                    self.stat_cols[task][c] = self.stat_cols[task][c][0]

        # mode specific inits
        if self.mode == 'trn_seg':
            self.curr_epoch = 0

            if 'save_check' not in stats_to_track['epoch']:
                # self.stat_cols[task]['save_check'] = 12                              # TODO: abstact away to any pls!
                self.stat_cols[task].insert(0, 'save_check')
                self.stat_indents[task].insert(0, 12)
        # endregion

        # create files
        if self.mode == 'trn_seg':
            for file_type in ['per_epoch_all.csv', 'per_epoch_checkpoints.csv', 'per_batch.csv', 'live.txt']:
                with open(self.dir / file_type, 'a') as file:

                    if file_type == 'live.txt':
                        file.writelines(('               curr     total    |     time elapsed      est. time remain    est. total time\n',
                                         # 'epoch       :\n',
                                         # 'trn batch   :\n',
                                         # 'val batch   :\n',
                                         '\n'))
                        pass

                    elif file_type == 'per_batch.csv':
                        titles = 'epoch #, dataset, batch #, '
                        task = 'batch'
                    elif file_type[:9] == 'per_epoch':
                        titles = 'epoch #, '
                        task = 'epoch'

                    for c, col in enumerate(self.stat_cols[task]):
                        titles += f"{col},{' ' * (self.stat_indents[task][c]  - len(col))}"
                    titles += 'time elapsed,     est. time remain,   est. total time\n'
                    file.writelines((titles, '\n'))
                                                                                                        # TODO: for batch & epochs, add in the total!
        elif self.mode == 'im_to_samp':
            raise NotImplementedError('TODO if this is not MATT looking at this ...very sorry')


    def track(self, seq, task):
        #add tdqm track
        for num, value in enumerate(seq):
            self.note(f'flag!!\t{num}\t{task}\n{self.stats}\n\n' * 25)
            if self.mode == 'trn_seg':

                # region timers
                if num == 0: # reset timer & avoid div0 errors
                    print('-'*40)
                    print('\t\t', 'STARTING\t', task)
                    print('-'*40)

                    self.start_time[task] = time.time()
                    time_curr = 0
                    time_total = 0
                    time_remain = 0
                else:
                    time_curr = time.time() - self.start_time[task]
                    time_total = time_curr / num * (len(seq)-1)
                    time_remain = (time_total - time_curr)
                # endregion

                # region per_batch.csv
                if task[4:] == 'batch' and num > 0 and len(self.stats['batch']) > 0:
                    row = f"{self.curr_epoch},{' '*(8 - len(str(self.curr_epoch)))}"   # TODO put in len(epoch) & len(task) in the csvs
                    row += f"{task[:3]},{' '*(8 - len(task[:3]))}"
                    row += f"{num-1},{' '*(8 - len(str(num-1)))}"

                    row = self.make_csv_row(task[4:],
                                            time_curr,
                                            time_remain,
                                            time_total,
                                            row=row)
                    with open(self.dir / 'per_batch.csv', 'a') as file:
                        file.write(row)

                    self.stats[task] = {}
                # endregion

                # region per_epoch.csv

                elif task == 'epoch' and num > 0 and len(self.stats['epoch']) > 0:

                    self.curr_epoch = num

                    row = f"{num-1},{' '*(8-len(str(num-1)))}"

                    row = self.make_csv_row(task,
                            time_curr,
                            time_remain,
                            time_total,
                            row=row)
                    with open(self.dir / 'per_epoch_all.csv', 'a') as file:
                        file.write(row)
                    # self.note('yeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeees??')
                    # self.note(self.stats)
                    if 'save_check' in self.stats[task]:
                        # self.note('YAYAYAYAYAYAYAYAYAYAYAYAYAYAYAYAYAYAYAYAYAYAYS')
                        with open(self.dir / 'per_epoch_checkpoints.csv', 'a') as file:
                            file.write(row)

                    self.stats[task] = {}
                # endregion

                # region add to live.txt file
                with open(self.dir / 'live.txt', 'r') as file:
                    data = file.readlines()

                EXISTS = False
                for d in range(len(data)):
                    
                    if data[d][:len(task)] == task:
                        index = d
                        EXISTS = True
                        break

                if not EXISTS:
                    index = len(data)
                    data.append('')

                time_remain = f"{(time_remain / 60):.0f}m " \
                              f"{(time_remain % 60):.0f}s"
                time_curr = f"{(time_curr / 60):.0f}m " \
                            f"{(time_curr % 60):.0f}s"
                time_total = f"{(time_total / 60):.0f}m " \
                             f"{(time_total % 60):.0f}s"
                data[index] = f"{task}{' '*(12-len(task))}:   " \
                              f"{num+1}{' '*(4-len(str(num+1)))} /   " \
                              f"{len(seq)}{' '*(4-len(str(len(seq))))}           " \
                              f"{time_curr}{' '*(12-len(time_curr))}         " \
                              f"{time_remain}{' '*(12-len(time_remain))}        " \
                              f"{time_total}{' '*(12-len(time_total))}        " \
                              f"\n"

                data = data[:index+1] # reset lower other vals for next time
                # TODO: just remove the curr value? do we need to delete all the vals - or try putting the '-> ' arrow at the start again

                # print(data[0], data[index])
                # self.print(data[0], data[index])

                with open(self.dir / 'live.txt', 'w') as file:
                    file.writelines(data)       # todo: except possible file writing errs!!!

                self.stats[task] = {}
                # endregion

            yield value

    def make_csv_row(self, # TODO: documentation!
                     task,
                     time_curr,
                     time_remain,
                     time_total,
                     row=''):
                                                                                                # TODO! it curr doesnt save the stats of the last in seq!!!
        # region add cols to row
        for c, col in enumerate(self.stat_cols[task]) :

            # self.note(task)
            # self.note(col)
            # self.note(self.stat_cols)
            # self.note(self.stat_cols[task])
            try:
                # self.note('---')
                # self.note(self.stats)
                # self.note(self.stats[task])
                # self.note('')
                # self.note('---')
                # self.note(self.stats[task][col])
                # self.note('---')
                stat = self.stats[task][col]
                try:
                    stat = f"{stat:.3f}"
                except (ValueError, TypeError):
                    stat = f"{stat}"                # self.stat_cols[task][col] => indent
                # self.note(self.stat_indents[task])
                # self.note(self.stat_indents[task][c])
                row += f"{stat},{' ' * (self.stat_indents[task][c] - len(stat))}"            # TODO: for batch & epochs, add in the total! (ex = if col=='total' len(seq))
            except KeyError: # if a stat has not been added for this row
                row += ',' + ' ' * self.stat_indents[task][c]                                # TODO: fix if they are too big, add space in column
        # endregion

        # region timers
        time_remain = f"{(time_remain / 60):.0f}m " \
                      f"{(time_remain % 60):.0f}s"
        time_curr = f"{(time_curr / 60):.0f}m " \
                    f"{(time_curr % 60):.0f}s"
        time_total = f"{(time_total / 60):.0f}m " \
                     f"{(time_total % 60):.0f}s"
        row += f"{time_curr},{' ' * (18 - len(time_curr))}" \
                f"{time_remain},{' ' * (18 - len(time_remain))}" \
                f"{time_total},{' ' * (18 - len(time_total))}\n"
        # endregion

        return row

    def add_stat(self, key, stat, task='epoch'):
        self.stats[task][key] = stat
    # TODO: when add_stat() check that these are part of the dict
    def add_stats(self, stats, task='epoch'): # todo: makes sure its a dict
        for key in stats:
            self.stats[task][key] = stats[key] # todo: can this be 'vectoried' or atleast not looped?

    def notify_end(self, task):
        print('-'*36)
        print('\t\t', task, '\tENDED')
        print('-'*36)

        if self.mode == 'trn_seg':

            # region timers
            time_curr = time.time() - self.start_time[task]
            time_total = time_curr
            time_remain = 0
            # endregion

            # region per_batch.csv
            if task == 'batch':
                with open(self.dir / 'per_batch.csv', 'r') as file:
                    last_line = file.readlines()[-1][:20].replace(' ', ',').split(',')
                    last_epoch, last_dataset, last_batch = [l for l in last_line if (l != ',') or (l != ' ')]

                row = f"{last_epoch},{' ' * (8 - len(str(last_epoch)))}"
                row += f"{last_dataset},{' ' * (8 - len(last_dataset))}"
                row += f"{last_batch},{' ' * (8 - len(str(last_batch)))}"

                row = self.make_csv_row(task,
                                        time_curr,
                                        time_remain,
                                        time_total,
                                        row=row)
                with open(self.dir / 'per_batch.csv', 'a') as file:
                    file.write(row)

                self.stats[task] = {}
            # endregion

            # region per_epoch.csv
            elif task == 'epoch':
                with open(self.dir / 'per_epoch.csv', 'r') as file:
                    last_epoch = file.readlines()[-1][:20].replace(' ', ',').split(',')[0]

                row = f"{last_epoch+1},{' ' * (8 - len(str(last_epoch+1)))}"

                row = self.make_csv_row(task,
                                        time_curr,
                                        time_remain,
                                        time_total,
                                        row=row)

                with open(self.dir / 'per_epoch_all.csv', 'a') as file:
                    file.write(row)
                if 'save_check' in self.stats:
                    with open(self.dir / 'per_epoch_checkpoints.csv', 'a') as file:
                        file.write(row)

                self.stats[task] = {}
            # endregion



    def note(self, line, end='\n'):
        # TODO: allow for printing things like epoch_num
        # TODO: allow more types other than just strings (with *arg), but what if *arg=['', []] then we need to go recursive! - look at
        with open(self.dir / 'notepad.txt', 'a') as file:
            file.write(f'{line}')
            file.write(end)

    # def print(self, *objects):
    #         # if style == 'thick panel':
    #     width = 0
    #     for obj in objects:
    #         if len(str(obj)) > width:
    #             width = len(str(obj))
    #     width += 8
    #     print('#' * width)
    #     # print('###' + ' '*(console_width-6) + '###')
    #     for obj in objects:
    #         obj = str(obj)
    #         buffer_size = (width - 8 - len(obj)) // 2
    #         print('###' + ' ' * buffer_size + obj + ' ' * (buffer_size + len(obj) % 2) + '###')
    #     # print('###' + ' '*(console_width-6) + '###')
    #     print('#' * width)
    #         # else:
    #         #     print(*objects)


if __name__ == '__main__':
    batches = [[0, 1, 2, 3, 55],
               [0, 1, 2, 3],
               [0, 1, 2, 3, 55, 33]]
    tracker = Tracking_Pane('C:/Users/muzwe/Documents/GitHub/geo-deep-learning - MATT/MATTS',
                            mode='trn_seg',
                            epoch_stats_to_track=['save_check'],
                            debug=True)
    for i in tracker.track(range(6), 'epoch'):#task_id=0):
        print('epoch', i)
        tracker.add_stat('lol', i)

        for j in tracker.track(batches[i % 3], 'trn batch'):#task_id=1):
            print('\ttrn batch :', j)
            time.sleep(1)
            for k in tracker.track(range(2), 'trn vis'):
                print('\t\ttrn vis :', k)
                time.sleep(1)

        for jj in tracker.track(batches[i % 3], 'val batch'):#task_id=1):
            print('\tval batch :', jj)
            time.sleep(1)
            for k in tracker.track(range(2), 'val vis'):
                print('\t\tval vis :', k)
                time.sleep(1)