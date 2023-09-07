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
        'office0'
        #   'office0_dense9995',→ what are these data?
          #'office1'
          #'office2',
          #'office3',
          #'office4'
          #'room0',
          #'room1'
          #'room2'
        ]

input_dir = '/scratch_net/biwidl215/myamaguchi/rpg_vid2e-master/data/Replica'

output_folder = '/scratch_net/biwidl215/myamaguchi/EvenNICER-SLAM/Datasets/replica_gt_events'



for place in places:
    list_of_image_files = sorted(glob.glob(f'{input_dir}/{place}/results/frame*.jpg'))
    n_frames = len(list_of_image_files)
    print('starting to process', place, 'with', n_frames, 'frames...\n')

    fps = 24 * 5 
    interval = 1 / fps

    list_of_timestamps = np.linspace(0, interval, num=2, endpoint=True)

    output_dir = os.path.join(output_folder, place)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(n_frames-1):
        txt_filename = place + '_' + 'events' + str(i).zfill(6) + '_' + str(i+1).zfill(6) + '.txt'
        events_list_of_images = esim.generateFromStampedImageSequence(
            list_of_image_files[i : i + 2], # list of absolute paths to images     
            list_of_timestamps + interval*i             # list of timestamps in ascending order
        ) 
    
        events_list_of_images_new = np.hstack((events_list_of_images[:, 2:3], events_list_of_images[:, 0:2], events_list_of_images[:, 3:]))
        events_list_of_images_new[:, 1:] = events_list_of_images_new[:, 1:].astype("int16")

        with open(os.path.join(output_dir, txt_filename), 'w') as f:
            f.write("1200 680\n")
            for row in events_list_of_images_new:
                line = " ".join(str(x) for x in row)
                f.write(line + '\n')


  

