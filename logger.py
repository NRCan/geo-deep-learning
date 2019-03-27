import os


class InformationLogger(object):
    def __init__(self, log_folder, mode):
        if mode == 'val':
            self.class_scores = open(os.path.join(log_folder, mode + "_classes_score.log"), "a")
            self.averaged_scores = open(os.path.join(log_folder, mode + "_averaged_score.log"), "a")
        self.losses_values = open(os.path.join(log_folder, mode + "_losses_values.log"), "a")

    def add_values(self, info, epoch, log_metrics=False):
        """Add new information to the logs."""
        self.losses_values.write(f"{epoch} {info['loss'].avg}\n")
        if log_metrics:
            self.averaged_scores.write(f"{epoch} {info['precision'].avg} {info['recall'].avg} {info['fscore'].avg}\n")
            del info['precision'], info['recall'], info['fscore'], info['loss']
            for key, value in info.items():
                self.class_scores.write(f"{epoch} {key} {info[key].avg}\n")
            self.class_scores.flush()
            self.averaged_scores.flush()
        self.losses_values.flush()


def save_logs_to_bucket(bucket, bucket_output_path, output_path, now, batch_metrics=None):
    if batch_metrics is not None:
        list_log_file = ['trn_losses_values', 'val_classes_score', 'val_averaged_score', 'val_losses_values']
    else:
        list_log_file = ['trn_losses_values', 'val_losses_values']
    for i in list_log_file:
        if bucket_output_path:
            log_file = os.path.join(output_path, f"{i}.log")
            bucket.upload_file(log_file, os.path.join(bucket_output_path, f"Logs/{now}_{i}.log"))
        else:
            log_file = os.path.join(output_path, f"{i}.log")
            bucket.upload_file(log_file, f"Logs/{now}_{i}.log")
    bucket.upload_file("output.txt", os.path.join(bucket_output_path, f"Logs/{now}_output.txt"))
