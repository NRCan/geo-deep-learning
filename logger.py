import os

class InformationLogger(object):
    def __init__(self, logFolder, mode):
        self.class_scores = open(os.path.join(logFolder, mode + "_classes_score.log"), "a")
        self.averaged_scores = open(os.path.join(logFolder, mode + "_averaged_score.log"), "a")
        self.losses_values = open(os.path.join(logFolder, mode + "_losses_values.log"), "a")
        
    def AddValues(self, info, epoch):

        self.class_scores.write(str(epoch))
        for x in range(0, len(info['prfScore'])):
            self.class_scores.write(str(' ' + str(info['prfScore'][x])))
        self.class_scores.write('\n')
        
        self.averaged_scores.write(str(str(epoch) + ' ' + str(info['prfAvg'][0]) + ' ' + str(info['prfAvg'][1]) + ' ' + str(info['prfAvg'][2]) + '\n'))
        self.losses_values.write(str(str(epoch) + ' ' + str(info['loss']) + '\n'))
        
    def EndLog(self):
        self.class_scores.close()
        self.averaged_scores.close()
        self.losses_values.close()
        
    def PlotTendencies(self):
        boubou = 1