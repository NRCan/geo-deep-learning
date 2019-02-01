import os


class InformationLogger(object):
    def __init__(self, log_folder, mode):
        self.class_scores = open(os.path.join(log_folder, mode + "_classes_score.log"), "a")
        self.averaged_scores = open(os.path.join(log_folder, mode + "_averaged_score.log"), "a")
        self.losses_values = open(os.path.join(log_folder, mode + "_losses_values.log"), "a")
        self.iou = open(os.path.join(log_folder, mode + "_iou.log"), "a")

    def add_values(self, info, epoch):
        """Add new information to the logs."""
        self.averaged_scores.write(str(str(epoch) + ' ' + str(info['precision'].avg) + ' ' + str(info['recall'].avg)
                                       + ' ' + str(info['fscore'].avg) + ' \n'))
        self.losses_values.write(str(str(epoch) + ' ' + str(info['loss'].avg) + '\n'))
        self.iou.write(str(str(epoch) + ' ' + str(info['iou'].avg) + '\n'))
        del info['precision'], info['recall'], info['fscore'], info['iou'], info['loss']
        for key, value in info.items():
            self.class_scores.write(str(epoch) + ' ' + str(key) + ' ' + str(info[key].avg) + '\n')
        self.class_scores.flush()
        self.averaged_scores.flush()
        self.losses_values.flush()
        self.iou.flush()

    def end_log(self):
        """Stop logging metrics."""
        self.class_scores.close()
        self.averaged_scores.close()
        self.losses_values.close()
        self.iou.close()


def save_logs_to_bucket(bucket, bucket_output_path, output_path, now):
    list_log_file = ['trn_classes_score', 'trn_averaged_score', 'trn_losses_values', 'trn_iou',
                     'val_classes_score', 'val_averaged_score', 'val_losses_values', 'val_iou']
    for i in list_log_file:
        if bucket_output_path:
            log_file = os.path.join(output_path, f"{i}.log")
            bucket.upload_file(log_file, os.path.join(bucket_output_path, f"Logs/{now}_{i}.log"))
        else:
            log_file = os.path.join(output_path, f"{i}.log")
            bucket.upload_file(log_file, f"Logs/{now}_{i}.log")
    bucket.upload_file("output.txt", os.path.join(bucket_output_path, f"Logs/{now}_output.txt"))
