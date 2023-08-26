import numpy as np
import cv2
import os
import glob
import esim_py

# constructor
contrast_threshold_pos = 0.1 # contrast thesholds for positive 
contrast_threshold_neg = 0.1 # and negative events
refractory_period = 1e-4 # minimum waiting period (in sec) before a pixel can trigger a new event
log_eps = 1e-3 # epsilon that is used to numerical stability within the logarithm
use_log = 1 # wether or not to use log intensity
esim = esim_py.EventSimulator(
    contrast_threshold_pos, 
    contrast_threshold_neg, 
    refractory_period, 
    log_eps, 
    use_log,  
    )

# setter, useful within a training loop
esim.setParameters(contrast_threshold_pos, contrast_threshold_neg, refractory_period, log_eps, use_log)

# generate Replica events
places = [
        #  'office0',
        #   'office0_dense9995',→ what are these data?
          'office1'
          #'office2',
          #'office3',
          #'office4',
          #'room0',
          #'room1',
          #'room2'
        ]

input_dir = '/scratch_net/biwidl215/myamaguchi/rpg_vid2e-master/data/Replica'

output_folder = '/scratch_net/biwidl215/myamaguchi/EvenNICER-SLAM/Datasets/replica_gt_events'



for place in places:
    list_of_image_files = sorted(glob.glob(f'{input_dir}/{place}/results/frame*.jpg'))
    # downsampling each image
    # list_of_downscaled_images = []
    # for image_file in list_of_image_files:
    #     image = cv2.imread(image_file)
    #     new_width = image.shape[1] // 4
    #     new_height = image.shape[0] // 4
    #     downscaled_image = cv2.resize(image, (new_width, new_height))
    #     output_file = image_file.replace('.jpg', f'_downsampled.jpg')
        #cv2.imwrite(output_file, downscaled_image)
        #list_of_downscaled_images.append(output_file)


    n_frames = len(list_of_image_files)
    print('starting to process', place, 'with', n_frames, 'frames...\n')

    fps = 24 * 5 
    interval = 1 / fps

    list_of_timestamps = np.linspace(0, interval, num=2, endpoint=True)

    output_dir = os.path.join(output_folder, place)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # events_list_of_images = esim.generateFromStampedImageSequence(
    #     list_of_image_files,
    #     list_of_timestamps
    # )
    # print(events_list_of_images.shape)

    #txt_filename = place + '_' + 'events.txt'
    #with open (os.path.join(output_dir, txt_filename), 'w') as f:
    for i in range(n_frames-1):
        txt_filename = place + '_' + 'events' + str(i).zfill(6) + '_' + str(i+1).zfill(6) + '.txt'
        events_list_of_images = esim.generateFromStampedImageSequence(
            list_of_image_files[i : i + 2],
            #list_of_downscaled_images[i : i+2],   # list of absolute paths to images   
            list_of_timestamps + interval*i             # list of timestamps in ascending order
        ) # each event: [x, y, t, polarity]
          # print(type(events_list_of_images)): numpy.ndarray
        events_list_of_images[:, :2] = events_list_of_images[:, :2].astype("int32")
        with open(os.path.join(output_dir, txt_filename), 'w') as f:
            for row in events_list_of_images:
                line = " ".join(str(x) for x in row)
                f.write(line + '\n')
        # timestamp, x, y, polarity

    # list_of_downsampled_files = glob.glob("/scratch_net/biwidl215/myamaguchi/rpg_vid2e-master/data/Replica/room0/results/*_downsampled.jpg")
    # for file in list_of_downsampled_files:
    #     os.remove(file)


