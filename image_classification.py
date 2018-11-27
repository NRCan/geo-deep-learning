import numpy as np
import os
import torch
import time
import argparse
import fnmatch
import warnings
from models.model_choice import net
from utils import read_parameters, create_new_raster_from_base, assert_band_number, load_from_checkpoint, \
    image_reader_as_array

try:
    import boto3
except ModuleNotFoundError:
    warnings.warn('The boto3 library counldn\'t be imported. Ignore if not using AWS s3 buckets', ImportWarning)
    pass


def main(bucket, work_folder, img_list, weights_file_name, model, number_of_bands, overlay):
    """Identify the class to which each image belongs.
    Args:
        bucket: bucket in which data is stored if using AWS S3
        work_folder: full file path of the folder containing images
        img_list: list containing images to classify
        weights_file_name: full file path of the file containing weights
        model: loaded model with which classification should be done
    """

    if torch.cuda.is_available():
        model = model.cuda()
    if bucket:
        bucket.download_file(weights_file_name, "saved_model.pth.tar")
        model = load_from_checkpoint("saved_model.pth.tar", model)
    # load weights
    else:
        model = load_from_checkpoint(weights_file_name, model)

    since = time.time()

    for img in img_list:
        if bucket:
            bucket.download_file(os.path.join(work_folder, img), "Images/"+img)
            assert_band_number("Images/"+img, number_of_bands)
        # assert that img band and the parameter in yaml have the same value
        else:
            assert_band_number(os.path.join(work_folder, img), number_of_bands)
        classification(bucket, work_folder, model, img, overlay)
        print('Image ', img, ' classified')
        if bucket:
            try:
                bucket.put_object(Key='Classified_Images/', Body='')
            except ClientError:
                pass
            classif_img = open('Classified_Images/'+img.split('.')[0] + '_classif.tif', 'rb')
            bucket.put_object(Key='Classified_Images/' + img.split('.')[0] + '_classif.tif', Body=classif_img)
    time_elapsed = time.time() - since
    print('Classification complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


def classification(bucket, folder_images, model, image, overlay):
    """Classify images
    Args:
        bucket: bucket in which data is stored if using AWS S3
        folder_images: full file path of the folder containing images
        model: model to use for classification
        image: image to classify
    """
    # Chunk size. Should not be modified often. We want the biggest chunk to be process at a time but,
    # a too large image chunk will bust the GPU memory when processing.
    chunk_size = 512

    # switch to evaluate mode
    model.eval()

    if bucket:
        input_image = image_reader_as_array("Images/"+image)
    else:
        input_image = image_reader_as_array(os.path.join(folder_images, image))
    if len(input_image.shape) == 3:
        h, w, nb = input_image.shape
        padded_array = np.pad(input_image, ((overlay, chunk_size), (overlay, chunk_size), (0, 0)), mode='constant')
    elif len(input_image.shape) == 2:
        h, w = input_image.shape
        padded_array = np.expand_dims(np.pad(input_image, ((overlay, chunk_size), (overlay, chunk_size)),
                                             mode='constant'), axis=0)
    else:
        h = 0
        w = 0
        padded_array = None

    output_np = np.empty([h + overlay + chunk_size, w + overlay + chunk_size, 1], dtype=np.uint8)

    if padded_array.any():
        with torch.no_grad():
            for row in range(0, h, chunk_size - (2 * overlay)):
                for col in range(0, w, chunk_size - (2 * overlay)):

                    chunk_input = padded_array[row:row + chunk_size, col:col + chunk_size, :]
                    inputs = torch.from_numpy(np.float32(np.transpose(chunk_input, (2, 0, 1))))

                    inputs.unsqueeze_(0)

                    if torch.cuda.is_available():
                        inputs = inputs.cuda()
                    # forward
                    outputs = model(inputs)

                    a, pred = torch.max(outputs, dim=1)
                    segmentation = torch.squeeze(pred)

                    row_from = row + overlay
                    row_to = row + chunk_size - overlay
                    col_from = col + overlay
                    col_to = col + chunk_size - overlay

                    useful_classification = segmentation[overlay:chunk_size - overlay, overlay:chunk_size - overlay]
                    output_np[row_from:row_to, col_from:col_to, 0] = useful_classification

            # Resize the output array to the size of the input image and write it
            output_np = output_np[overlay:h + overlay, overlay:w + overlay]
            if bucket:
                create_new_raster_from_base("Images/"+image, "Classified_Images/"+image.split('.')[0] + '_classif.tif',
                                            1, output_np)
            else:
                create_new_raster_from_base(os.path.join(folder_images, image),
                                            os.path.join(folder_images, image.split('.')[0] + '_classif.tif'), 1,
                                            output_np)
    else:
        print("Error classifying image : Image shape of {:1} is not recognized".format(len(input_image.shape)))


if __name__ == '__main__':
    print('Start: ')
    parser = argparse.ArgumentParser(description='Image classification using trained model')
    parser.add_argument('param_file', metavar='file',
                        help='Path to training parameters stored in yaml')
    args = parser.parse_args()
    bucket = None
    params = read_parameters(args.param_file)
    working_folder = params['classification']['working_folder']

    model, sdp, model_depth = net(params, rtn_level=True)
    nbr_pix_overlay = 2 ** (model_depth + 1)

    bucket_name = params['global']['bucket_name']

    if bucket_name:
        list_img = []
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucket_name)
        for f in bucket.objects.filter(Prefix=working_folder):
            if f.key != working_folder + '/':
                list_img.append(f.key.split('/')[-1])
    else:
        list_img = [img for img in os.listdir(working_folder) if fnmatch.fnmatch(img, "*.tif*")]
    main(bucket, params['classification']['working_folder'], list_img, params['classification']['state_dict_path'],
         model, params['global']['number_of_bands'], nbr_pix_overlay)

