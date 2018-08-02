import os

class InformationLogger(object):
    def __init__(self, logFolder, mode):
        self.class_scores = open(os.path.join(logFolder, mode + "_classes_score.log"), "a")
        self.averaged_scores = open(os.path.join(logFolder, mode + "_averaged_score.log"), "a")
        self.losses_values = open(os.path.join(logFolder, mode + "_losses_values.log"), "a")
        self.iou = open(os.path.join(logFolder, mode + "_iou.log"), "a")

    def AddValues(self, info, epoch):

        self.averaged_scores.write(str(str(epoch) + ' ' + str(info['precision'].avg) + ' ' +  str(info['recall'].avg) + ' ' + str(info['fscore'].avg) + ' \n'))
        self.losses_values.write(str(str(epoch) + ' ' + str(info['loss'].avg) + '\n'))
        self.iou.write(str(str(epoch) + ' ' + str(info['iou'].avg) + '\n'))

        del info['precision'], info['recall'], info['fscore'], info['iou'], info['loss']

        for key, value in info.items():
            self.class_scores.write(str(epoch) + ' ' + str(key) + ' ' + str(info[key].avg) + '\n')

        self.class_scores.flush()
        self.averaged_scores.flush()
        self.losses_values.flush()
        self.iou.flush()

    def EndLog(self):
        self.class_scores.close()
        self.averaged_scores.close()
        self.losses_values.close()
        self.iou.close()
