import torch
# import torch should be first. Unclear issue, mentionned here: https://github.com/pytorch/pytorch/issues/2083
import numpy as np
import os
import csv
import time
import argparse
import heapq
import rasterio
from PIL import Image
import torchvision
from models.model_choice import net, maxpool_level
from utils import read_parameters, assert_band_number, load_from_checkpoint, \
    image_reader_as_array, read_csv

try:
    import boto3
except ModuleNotFoundError:
    pass


def main(bucket, work_folder, img_list, weights_file_name, model, number_of_bands, overlay, classify, num_classes):
    """Identify the class to which each image belongs.
    Args:
        bucket: bucket in which data is stored if using AWS S3
        work_folder: full file path of the folder containing images
        img_list: list containing images to classify
        weights_file_name: full file path of the file containing weights
        model: loaded model with which inference should be done
        number_of_bands: number of bands in the input rasters
        overlay: amount of overlay to apply
        classify: True if doing a classification task, False if doing semantic segmentation
    """
    if torch.cuda.is_available():
        model = model.cuda()
    if bucket:
        bucket.download_file(weights_file_name, "saved_model.pth.tar")
        model = load_from_checkpoint("saved_model.pth.tar", model)
        if classify:
            classes_file = weights_file_name.split('/')[:-1]
            class_csv = ''
            for folder in classes_file:
                class_csv = os.path.join(class_csv, folder)
            bucket.download_file(os.path.join(class_csv, 'classes.csv'), 'classes.csv')
            with open('classes.csv', 'rt') as file:
                reader = csv.reader(file)
                classes = list(reader)
    else:
        model = load_from_checkpoint(weights_file_name, model)
        if classify:
            classes_file = weights_file_name.split('/')[:-1]
            class_path = ''
            for c in classes_file:
                class_path = class_path + c + '/'
            with open(class_path + 'classes.csv', 'rt') as f:
                reader = csv.reader(f)
                classes = list(reader)
    since = time.time()
    classified_results = np.empty((0, 2 + num_classes))

    for img in img_list:
        img_name = os.path.basename(img['tif'])
        if bucket:
            local_img = f"Images/{img_name}"
            bucket.download_file(img['tif'], local_img)
            inference_image = f"Classified_Images/{img_name.split('.')[0]}_inference.tif"
        else:
            local_img = img['tif']
            inference_image = os.path.join(work_folder, f"{img_name.split('.')[0]}_inference.tif")

        assert_band_number(local_img, number_of_bands)
        if classify:
            outputs, predicted = classifier(bucket, model, img['tif'])
            top5 = heapq.nlargest(5, outputs.cpu().numpy()[0])
            top5_loc = []
            for i in top5:
                top5_loc.append(np.where(outputs.cpu().numpy()[0] == i)[0][0])
            print(f"Image {img_name} classified as {classes[0][predicted]}")
            print('Top 5 classes:')
            for i in range(0, 5):
                print(f"\t{classes[0][top5_loc[i]]} : {top5[i]}")
            classified_results = np.append(classified_results, [np.append([img['tif'], classes[0][predicted]],
                                                                          outputs.cpu().numpy()[0])], axis=0)
            print()
        else:
            sem_seg_results = sem_seg_inference(bucket, model, img['tif'], overlay)
            create_new_raster_from_base(local_img, inference_image, sem_seg_results)
            print(f"Semantic segmentation of image {img_name} completed")

        if bucket:
            if not classify:
                bucket.upload_file(inference_image, os.path.join(work_folder, f"{img_name.split('.')[0]}_inference.tif"))

    if classify:
        csv_results = 'classification_results.csv'
        if bucket:
            np.savetxt(csv_results, classified_results, fmt='%s', delimiter=',')
            bucket.upload_file(csv_results, os.path.join(work_folder, csv_results))
        else:
            np.savetxt(os.path.join(work_folder, csv_results), classified_results, fmt='%s', delimiter=',')

    time_elapsed = time.time() - since
    print('Inference completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


def create_new_raster_from_base(input_raster, output_raster, write_array):
    """Function to use info from input raster to create new one.
    Args:
        input_raster: input raster path and name
        output_raster: raster name and path to be created with info from input
        write_array (optional): array to write into the new raster

    Return:
        none
    """

    with rasterio.open(input_raster, 'r') as src:
        with rasterio.open( output_raster, 'w',
                            driver=src.driver,
                            width=src.width,
                            height=src.height,
                            count=1,
                            crs=src.crs,
                            dtype=np.uint8,
                            transform=src.transform) as dst:
            dst.write(write_array[:,:,0], 1)


def sem_seg_inference(bucket, model, image, overlay):
    """Inference on images using semantic segmentation
    Args:
        bucket: bucket in which data is stored if using AWS S3
        model: model to use for inference
        image: full path of the image to infer on
        overlay: amount of overlay to apply

        returns a numpy array of the same size (h,w) as the input image, where each value is the predicted output.
    """
    # Chunk size. Should not be modified often. We want the biggest chunk to be process at a time but,
    # a too large image chunk will bust the GPU memory when processing.
    chunk_size = 512

    # switch to evaluate mode
    model.eval()

    if bucket:
        input_image = image_reader_as_array(f"Images/{os.path.basename(image)}")
    else:
        input_image = image_reader_as_array(image)

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

                    useful_sem_seg = segmentation[overlay:chunk_size - overlay, overlay:chunk_size - overlay]
                    output_np[row_from:row_to, col_from:col_to, 0] = useful_sem_seg

            # Resize the output array to the size of the input image and write it
            output_np = output_np[overlay:h + overlay, overlay:w + overlay]
            return output_np
    else:
        print("Error classifying image : Image shape of {:1} is not recognized".format(len(input_image.shape)))


def classifier(bucket, model, image):
    """Classify images by class
        Args:
            bucket: bucket in which data is stored if using AWS S3
            model: model to use for classification
            image: image to classify
        """
    model.eval()
    if bucket:
        img = Image.open(f"Images/{os.path.basename(image)}").resize((299, 299), resample=Image.BILINEAR)
    else:
        img = Image.open(image).resize((299, 299), resample=Image.BILINEAR)
    to_tensor = torchvision.transforms.ToTensor()
    img = to_tensor(img)
    img = img.unsqueeze(0)
    with torch.no_grad():
        if torch.cuda.is_available():
            img = img.cuda()
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
    return outputs, predicted


if __name__ == '__main__':
    print('Start: ')
    parser = argparse.ArgumentParser(description='Inference on images using trained model')
    parser.add_argument('param_file', metavar='file',
                        help='Path to training parameters stored in yaml')
    args = parser.parse_args()
    bucket = None
    params = read_parameters(args.param_file)
    working_folder = params['inference']['working_folder']
    csv_file = params['inference']['img_csv_file']

    if params['global']['classify']:
        model, sdp = net(params)
        model_depth = maxpool_level(model, params['global']['number_of_bands'], 299)['MaxPoolCount']
    else:
        model, sdp, model_depth = net(params, rtn_level=True)
    nbr_pix_overlay = 2 ** (model_depth + 1)

    bucket_name = params['global']['bucket_name']

    if bucket_name:
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucket_name)
        bucket.download_file(csv_file, 'img_csv_file.csv')
        list_img = read_csv('img_csv_file.csv', inference=True)
    else:
        list_img = read_csv(csv_file, inference=True)
    main(bucket,
         params['inference']['working_folder'],
         list_img,
         params['inference']['state_dict_path'],
         model,
         params['global']['number_of_bands'],
         nbr_pix_overlay,
         params['global']['classify'],
         params['global']['num_classes'])

